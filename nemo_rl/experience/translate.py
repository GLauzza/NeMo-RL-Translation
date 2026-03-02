import re

from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec


def chunk(text, tokenizer, chunk_size):
    print("input text:", text)
    splitted = re.split(r"((?:(?<![\.:])[\.\?\!\n][\s+\n]\n*)(?!\s*-))", text)
    sentences = (
        [chunk + sep for chunk, sep in zip(splitted[0::2], splitted[1::2])] +
        [splitted[-1]]
    )

    chunks = []
    chunk_length = 0
    for sentence in sentences:
        n_tokens = tokenizer(sentence, return_length=True)["length"][0]
        if chunk_length + n_tokens > chunk_size:
            chunk_length = 0
        if chunk_length == 0:
            chunks.append(sentence)
        else:
            chunks[-1] += sentence
        chunk_length += n_tokens
    
    chunks_seps = []
    for i, chunk in enumerate(chunks):
        stripped_chunk = chunk.rstrip()
        chunks_seps.append(chunk[len(stripped_chunk):])
        chunks[i] = stripped_chunk
    
    print("output chunks", chunks)
    return chunks, chunks_seps

 
def chunk_batch(texts, tokenizer, input_name, chunk_size):
    print("FM - Chunking Data")
    chunked = []
    chunked_seps = []
    for text in texts:
        chunks, chunks_seps = chunk(text, tokenizer, chunk_size)
        chunked.append(chunks)
        chunked_seps.append(chunks_seps)

    flattened_chunks = [chunk for chunks in chunked for chunk in chunks]
    flattened_chunks_seps = [chunk_sep for chunks_seps in chunked_seps for chunk_sep in chunks_seps]
    sample_ids = [i for i, chunks in enumerate(chunked) for chunk in chunks]
    chunk_ids = [i for chunks in chunked for i, chunk in enumerate(chunks)]

    dataset = Dataset.from_dict({
        input_name: flattened_chunks,
        "sep": flattened_chunks_seps,
        "sample_id": sample_ids,
        "chunk_id": chunk_ids,
        "id": list(range(len(flattened_chunks))),
    })
    print("FM - Chunked Data")
    return dataset


def prepare_inference_data(dataset, tokenizer, batch_size=-1, input_name="question", use_only_input=False, sortby=None, chunk_size=-1):
    print("FM - Preparing Data")
    if chunk_size == -1:
        dataset, dataloader, sources = prepare_sorted_inference_data(dataset, tokenizer, batch_size, input_name, use_only_input, sortby)
        print("FM - Prepared Data")
        return dataset, dataloader, sources

    chunked_dataset = chunk_batch(dataset[input_name], tokenizer, input_name, chunk_size)

    chunk_n_data = chunked_dataset.filter(lambda x : x["chunk_id"] == 0)
    _, dataloader, _ = prepare_sorted_inference_data(chunk_n_data, tokenizer, batch_size, input_name)

    print("FM - Prepared Data")
    return chunked_dataset, dataloader, None


def prepare_sorted_inference_data(dataset, tokenizer, batch_size=-1, input_name="question", use_only_input=False, sortby=None, answer_start=None):
    if sortby is None:
        sortby = input_name
    conversations = [
        [{
            "role": "user", 
            "content": (
                "Please translate sentence by sentence the full following text in French.\n"
                "- Only output the translation.\n"
                "- Only translate what is after <|Text|>.\n"
                # "- Don't summarize.\n"
                "- Don't solve the problem, only translate.\n"
                "- Preserve any mathematical formula formatting.\n"
                "- Don't translate what is inside \\boxed{}.\n"
                "<|Text|>\n"
                f"{input_field}"
            )
        }]
        for input_field in dataset[input_name]
    ]
    if answer_start is None:
        dataset = dataset.add_column(
            "chat_input",
            [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in conversations]
        )
    else:
        dataset = dataset.add_column(
            "chat_input",
            [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + answer for messages, answer in zip(conversations, answer_start)]
        )
    dataset = dataset.add_column(
        "length",
        [len(x) for x in dataset[sortby]]
    )
    dataset = dataset.sort("length")

    if batch_size == -1:
        batch_size = len(dataset)
    if use_only_input:
        dataloader = DataLoader(dataset["chat_input"], batch_size=batch_size, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sources = None
    if "source" in dataset.column_names:
        sources = set(dataset["source"])

    return dataset, dataloader, sources


def chat_input_to_generation_input_data(texts):
    tokenized_texts = tokenizer(texts, padding=True, return_tensors="pt", return_length=True)
    generation_input_data = BatchedDataDict[GenerationDatumSpec]({"input_ids": tokenized_texts["input_ids"], "input_lengths": tokenized_texts["length"]})        
    return generation_input_data


def generation_outputs_to_generated_texts(generation_outputs):
    # Extract everything we need from the generation outputs
    output_ids = generation_outputs["output_ids"]
    generation_lengths = generation_outputs["generation_lengths"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]

    generated_ids = []
    for i in range(len(input_lengths)):
        input_len = input_lengths[i].item()
        total_length = unpadded_sequence_lengths[i].item()
        full_output = output_ids[i]
        generated_part = full_output[input_len:total_length]
        generated_ids.append(generated_part)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


def infer_chunked(policy_generation, raw_dataset, chunked_dataset, dataloader, output_name, sampling_params, batch_size, input_name, chunk_size):

    max_chunks = max(chunked_dataset["chunk_id"])
    n_sample = max(chunked_dataset["sample_id"]) + 1
    inputs = [""] * n_sample
    outputs = [[]] * n_sample
    for i in tqdm(range(max_chunks)):
        print(f"FM - Infering chunk {i}/{max_chunks}")
        for data in tqdm(dataloader):
            # print(f"\n\n\nLens:{[len(sample) for sample in data['chat_input']]}\n\nInputs:\n{data['chat_input']}\n\n")
            generation_outputs = policy_generation.generate(chat_input_to_generation_input_data(data["chat_input"]), greedy=greedy)
            output = generation_outputs_to_generated_texts(generation_outputs)
            output_lens = generation_outputs["generation_lengths"]
            for sample_id, inp, out, out_len, sep in zip(data["sample_id"], data[input_name], output, output_lens, data["sep"]):
                if out_len < 1.75*chunk_size and inputs[sample_id] != "<DISCARDED>":
                    if i == 0:
                        outputs[sample_id] = [out + sep]
                    else:
                        outputs[sample_id].append(out + sep)
                    inputs[sample_id] = inp + sep
                else:
                    if i == 0:
                        outputs[sample_id] = "<DISCARDED>"
                    else:
                        outputs[sample_id].append("<DISCARDED>")
                    inputs[sample_id] = "<DISCARDED>"
            # print(f"\n\n\nLens:{[len(sample) for sample in output]}\n\nOutputs:\n{output}\n\n\n")

        if i < max_chunks:
            chunk_n_data = chunked_dataset.filter(lambda x : x["chunk_id"] == i+1)
            chunk_n_data = chunk_n_data.add_column(
                "concatenated_chunks",
                [inputs[sample["sample_id"]] + sample[input_name] for sample in chunk_n_data]
            )
            chunk_n_data = chunk_n_data.filter(lambda x : not x["concatenated_chunks"].startswith("<DISCARDED>"))
            answer_start = [outputs[sample_id][i] for sample_id in chunk_n_data["sample_id"]]
            if len(chunk_n_data) == 0:
                break
            _, dataloader, _ = prepare_sorted_inference_data(chunk_n_data, tokenizer, batch_size=batch_size, input_name="concatenated_chunks", answer_start=answer_start)
        
    outputs = [" ".join(output) for output in outputs]
    print("FM - Infered")

    raw_dataset = raw_dataset.add_column(output_name, outputs)
    return raw_dataset


def translate(generated_texts, tokenizer, batch_size=-1, chunk_size=512, input_name="solution", output_name="solution_fr"):
    raw_dataset = Dataset.from_dict({input_name:text for text in generated_texts})

    dataset, dataloader, _ = prepare_inference_data(
        raw_dataset,
        tokenizer,
        batch_size=batch_size,
        input_name=input_name,
        use_only_input=True,
        sortby=input_name,
        chunk_size=chunk_size,
    )

    dataset = infer_chunked(policy_generation, raw_dataset, dataset, dataloader, output_name, sampling_params, batch_size, input_name, chunk_size)

    raise Exception("test")