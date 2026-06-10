import re
import os
import time

import torch
from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
import fasttext

import gc
import torch
# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec, GenerationOutputSpec
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.vllm import VllmConfig




TRANSLATE_PROMPT = (
    # "Please translate sentence by sentence the full following text in French.\n"
    # "- Only output the translation.\n"
    # # "- Only translate what is after <|Text|>.\n"
    # # "- Don't summarize.\n"
    # "- Don't solve the problem, only translate.\n"
    # "- Preserve any mathematical formula formatting.\n"
    # # "- Don't translate what is inside \\boxed{}.\n"
    # # "<|Text|>\n"
    """You are a professional translator. Translate the following text into French. Preserve all mathematical expressions, symbols, and formatting exactly as they appear in the original text. Do not modify any numbers, equations, or formulas. Translate all mathematical commands into first-person plural.

Text:
{text}
"""
)


def remove_math(text):
    letters = "a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ"
    # remove latex and math content
    text = re.sub(r"\$\$(.*?)\$\$", " ", text)
    text = re.sub(r"\$(.*?)\$", " ", text)    
    text = re.sub(r"\\\[(.*?)\\\]", " ", text)
    text = re.sub(r"\\\{(.*?)\\\}", " ", text)    
    text = re.sub(r"\{(.*?)\}", " ", text)
    text = re.sub(r"\[(.*?)\]", " ", text)
    text = re.sub(r"\((.*?)\)", " ", text)
    text = re.sub(rf"\\[{letters}]+\{{.*?\}}", " ", text)
    text = re.sub(rf"\\[{letters}]+", " ", text)
    # ... -> .
    text = re.sub(r"\s*\.\s*\.\s*\.\s*", ". ", text)
    # remove words containing non-words
    text = re.sub(rf"\b\w*[^{letters}0-9\.\s'’:\?\!,;-]+\w*\b", " ", text)
    # remove special chars
    text = re.sub(rf"[^{letters}\.\s'’:\?\!,;]", " ", text, flags=re.UNICODE)
    # remove isolated single character
    for _ in range(2):
        text = re.sub(rf"\b[{letters}0-9]['’]*\b(?:\s+\b[{letters}0-9]['’]*\b)+", " ", text)
        text = re.sub(rf"\s+[^{letters}0-9:\?\!;]\s+", " ", text)
        text = re.sub(r"\W(?:\s+\W)+", " ", text)
    # remove isolated numbers
    text = re.sub(r"(?:\s+(?:(?:mod)?[0-9]+[\.,]?)+){2,}\s+", " ", text)
    # strip
    text = re.sub(r"\s+", " ", text).strip()
    # remove repeated words
    text = re.sub(rf"\b((?:[{letters}0-9]+\s*){{1,5}})\b(?:\s+\1)+", r"\1", text)
    return text


def is_french(model, text):
    non_math_text = remove_math(text).split("\n")
    output = model.predict(non_math_text, k=5)
    unique_langs = set([lang for langs in output[0] for lang in langs])
    languages = {unique_lang:0 for unique_lang in unique_langs}
    for langs, probs in zip(output[0], output[1]):
        for lang, prob in zip(langs, probs):
            languages[lang] += prob
    languages = {lang:(prob/len(non_math_text)) for lang, prob in languages.items()}
    fr_percent = languages.get("__label__fra_Latn",0)
    print(f"French percentage: {fr_percent}")
    return fr_percent > 0.98

def chunk(text, tokenizer, chunk_size):
    # print("input text:", text)
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
    
    # print("output chunks", chunks)
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
        [
            # {"role": "system", "content": TRANSLATE_PROMPT},
            # {"role": "user", "content": TRANSLATE_PROMPT + input_field},
            {"role": "user", "content": TRANSLATE_PROMPT.format(text=input_field)},
            
        ]
        for input_field in dataset[input_name]
    ]
    if answer_start is None:
        dataset = dataset.add_column(
            "chat_input",
            [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) for messages in conversations]
        )
    else:
        dataset = dataset.add_column(
            "chat_input",
            [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) + answer for messages, answer in zip(conversations, answer_start)]
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


def chat_input_to_generation_input_data(texts, tokenizer):
    tokenized_texts = tokenizer(texts, padding=True, return_tensors="pt", return_length=True)
    generation_input_data = BatchedDataDict[GenerationDatumSpec]({"input_ids": tokenized_texts["input_ids"], "input_lengths": tokenized_texts["length"]})        
    return generation_input_data


def generation_outputs_to_generated_texts(generation_outputs, tokenizer):
    # Extract everything we need from the generation outputs
    output_ids = generation_outputs["output_ids"]
    generation_lengths = generation_outputs["generation_lengths"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]
    input_lengths = [unpadded_sequence_length - generation_length for generation_length, unpadded_sequence_length in zip(generation_lengths, unpadded_sequence_lengths)]

    generated_ids = []
    for i in range(len(generation_lengths)):
        input_len = input_lengths[i].item()
        total_length = unpadded_sequence_lengths[i].item()
        full_output = output_ids[i]
        generated_part = full_output[input_len:total_length]
        generated_ids.append(generated_part)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


def infer_chunked(policy_generation, tokenizer, model_id, greedy, raw_dataset, chunked_dataset, dataloader, output_name, batch_size, input_name, chunk_size, discard_ratio):
    max_chunks = max(chunked_dataset["chunk_id"])
    n_sample = max(chunked_dataset["sample_id"]) + 1
    inputs = [""] * n_sample
    outputs = [[]] * n_sample

    start_time = time.time()
    policy_generation = LLM(
        model_id,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=4,
    )
    # vllm_cfg = policy_generation.cfg
    # vllm_cfg["vllm_cfg"]["gpu_memory_utilization"] = 0.75
    # policy_generation = VllmGeneration(
    #     RayVirtualCluster(
    #         name="grpo_inference_cluster",
    #         bundle_ct_per_node_list=[4] * 2,
    #         use_gpus=True,
    #         num_gpus_per_node=4,
    #         max_colocated_worker_groups=1,
    #     ),
    #     vllm_cfg,
    # )
    print(f"Translation model loading took {time.time() - start_time}s")

    
    for i in tqdm(range(max_chunks)):
        print(f"FM - Infering chunk {i}/{max_chunks}")
        for data in tqdm(dataloader):
            # print(f"\n\n\nLens:{[len(sample) for sample in data['chat_input']]}\n\nInputs:\n{data['chat_input']}\n\n")
            print(i)
            
            request_outputs = policy_generation.generate(data["chat_input"], SamplingParams(n=1, temperature=0.7, top_p=0.95, max_tokens=int(discard_ratio*chunk_size)))
            output = [request_output.outputs[0].text for request_output in request_outputs]
            output_lens = [len(request_output.outputs[0].token_ids) for request_output in request_outputs]

            # generation_input_data = chat_input_to_generation_input_data(data["chat_input"], tokenizer)
            # generation_outputs = policy_generation.generate(generation_input_data, greedy=greedy, max_new_tokens=discard_ratio*chunk_size)
            # output = generation_outputs_to_generated_texts(generation_outputs, tokenizer)
            # output_lens = generation_outputs["generation_lengths"]

            # print(f'\ninput: {data["chat_input"][0]}\nlength: {output_lens[0]}\n output: {output[0]}')
            for sample_id, inp, out, out_len, sep, chat_input in zip(data["sample_id"], data[input_name], output, output_lens, data["sep"], data["chat_input"]):
                inp_len = tokenizer(chunked_dataset.filter(lambda x : x["sample_id"] == sample_id and x["chunk_id"] == i)[input_name][0], padding=False, return_tensors="pt", return_length=True)["length"][0]
                # print(f"\n\n\nGENERATION {i}/{max_chunks} of id {sample_id} of len {out_len} compared to {inp_len} -------------------------------------------------\n\nINPUT--------:\n\n{chat_input}\n\nOUTPUT--------:\n\n{out}")
                if out_len < (discard_ratio*inp_len) and out_len > (discard_ratio/inp_len) and inputs[sample_id] != "<DISCARDED>":
                    if i == 0:
                        outputs[sample_id] = [out + sep]
                    else:
                        outputs[sample_id].append(out + sep)
                    inputs[sample_id] = inp + sep
                else:
                    print("<<<<<<<<<<<<<<<<<<<<<<<<<<DISCARDED>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    if i == 0:
                        outputs[sample_id] = ["<DISCARDED>"]
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
    # [print(f"\n\n\n\n\n\nOUTPUUUUUUUUUUUUUUUUUUUUUUUUUUUT {i}----------" ,output) for i, output in enumerate(outputs)]
    print("FM - Infered")

    # destroy_model_parallel()
    # del policy_generation.llm_engine.driver_worker
    os.system("nvidia-smi")
    del policy_generation
    gc.collect()
    torch.cuda.empty_cache()
    # torch.distributed.destroy_process_group()
    os.system("nvidia-smi")

    raw_dataset = raw_dataset.add_column(output_name, outputs)
    return raw_dataset


def translate(generated_texts, policy_generation, tokenizer, greedy, generation_outputs, input_lengths, max_seq_len, batch_size=-1, chunk_size=256, discard_ratio=1.6, input_name="solution", output_name="solution_fr"):
    os.system("nvidia-smi")

    model_id = os.environ.get("NEMO_RL_TRANSLATION_MODEL")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    fasttext_model = fasttext.load_model(os.environ.get("DSDIR")+"/HuggingFace_Models/facebook/fasttext-language-identification/model.bin")

    dataset = Dataset.from_dict({input_name:[text for text in generated_texts]})

    dataset_dict = {"id":[], "type": [], "is_french": [], input_name: []}

    i = 0
    original_is_french = []
    for sample in dataset:
        think_part = sample[input_name].split("<think>")[-1].split("</think>")[0].strip()
        dataset_dict["id"].append(i)
        dataset_dict["type"].append("think")
        is_think_french = is_french(fasttext_model, think_part)
        dataset_dict["is_french"].append(is_think_french)
        dataset_dict[input_name].append(think_part)
        original_is_french.append(is_think_french)
        i += 1
        if sample[input_name].count("</think>") > 0:
            answer_part = sample[input_name].split("</think>")[-1].strip()
            dataset_dict["id"].append(i)
            dataset_dict["type"].append("answer")
            is_answer_french = is_french(fasttext_model, answer_part)
            dataset_dict["is_french"].append(is_answer_french)
            dataset_dict[input_name].append(answer_part)
            original_is_french[-1] = original_is_french[-1] and is_answer_french
            i += 1

    dataset = Dataset.from_dict(dataset_dict)
    english_dataset = dataset.filter(lambda x: not x["is_french"])

    chunked_dataset, dataloader, _ = prepare_inference_data(
        english_dataset,
        tokenizer,
        batch_size=batch_size,
        input_name=input_name,
        use_only_input=True,
        sortby=input_name,
        chunk_size=chunk_size,
    )

    # dataset = Dataset.from_dict({input_name:[text for text in generated_texts], output_name:[text for text in generated_texts]})
    start_time = time.time()
    english_dataset = infer_chunked(policy_generation, tokenizer, model_id, greedy, english_dataset, chunked_dataset, dataloader, output_name, batch_size, input_name, chunk_size, discard_ratio)
    print(f"VLLM translation took {time.time() - start_time}s")
    # for i, sample in enumerate(dataset):
    #     print(f"SAMPLE: {i}", sample[output_name][:100])
    def recompute_is_french(sample):
        sample["is_french"] = is_french(fasttext_model, sample[output_name])
        return sample
    english_dataset = english_dataset.map(recompute_is_french)

    merged_dataset_dict = {"text": [], "is_french": []}
    for i in range(len(dataset)):
        if len(english_dataset.filter(lambda x: x["id"] == i)) > 0:
            sample = english_dataset.filter(lambda x: x["id"] == i)[0]
            text_column = output_name
        else:
            sample = dataset.filter(lambda x: x["id"] == i)[0]
            text_column = input_name
        if sample["type"] == "think":
            merged_dataset_dict["text"].append("<think>\n\n"+sample[text_column])
            merged_dataset_dict["is_french"].append(sample["is_french"])
        elif sample["type"] == "answer":
            merged_dataset_dict["text"][-1] += "</think>\n\n"+sample[text_column]
            merged_dataset_dict["is_french"][-1] = merged_dataset_dict["is_french"][-1] and sample["is_french"]

    dataset = Dataset.from_dict(merged_dataset_dict)
    translated_is_french = dataset["is_french"]

    new_generation_lengths = []
    new_unpadded_sequence_lengths = []
    for i ,(sample, input_length) in enumerate(zip(dataset, input_lengths)):
        if "<DISCARDED>" in sample["text"]:
            print(f"DISCARDED {i}---------------------------------------------------------------------------")
            new_generation_lengths.append(max_seq_len-input_length)
        else:
            generation_length = tokenizer(sample["text"], return_length=True)["length"][0]
            new_generation_lengths.append(min(generation_length, max_seq_len-input_length))
        new_unpadded_sequence_lengths.append(min(input_length + new_generation_lengths[-1], max_seq_len))
    max_len = max(new_unpadded_sequence_lengths + [len(generation_outputs["output_ids"][0])])
    
    new_output_ids = []
    for sample, output_ids, input_length in zip(dataset, generation_outputs["output_ids"], input_lengths):
        if "<DISCARDED>" in sample["text"]:
            new_output_ids.append(output_ids[:input_length].tolist() + [tokenizer(" discarded")["input_ids"][0]]*(max_len-input_length))
            print(f"Discarded is len {len(new_output_ids[-1])} {input_length}, {max_len}, {max_seq_len}")
        else:
            ids = (output_ids[:input_length].tolist() + tokenizer(sample["text"])["input_ids"])[:max_seq_len]
            new_output_ids.append(ids + [0]*(max_len-len(ids)))
            print(f"Non discarded is len {len(new_output_ids[-1])} {len(ids)}, {max_len}, {max_seq_len}")


    # Not the right logprobs, should have to recompute because prompt is different
    # new_logprobs = []
    # for logprob, input_length in zip(logprobs, input_lengths):
    #     filtered_logprob = [0]*input_length + [l[0] for l in logprob if l[0] !=0]
    #     new_logprobs.append(filtered_logprob + [0]*(max_len-len(filtered_logprob)))

    generation_outputs_translation = BatchedDataDict[GenerationOutputSpec]({
        "output_ids": torch.tensor(new_output_ids),
        "generation_lengths": torch.tensor(new_generation_lengths),
        "unpadded_sequence_lengths": torch.tensor(new_unpadded_sequence_lengths),
        # "logprobs": torch.tensor(new_logprobs),
        "truncated": None,
    })

    correct_length_output_ids = []
    # correct_length_logprobs = []
    for output_ids, logprob in zip(generation_outputs["output_ids"], generation_outputs["logprobs"]):
        correct_length_output_ids.append(output_ids.tolist() + [0]*(max_len-len(output_ids)))
        # correct_length_logprobs.append(logprob.tolist() + [0]*(max_len-len(logprob)))

    generation_outputs["output_ids"] = torch.tensor(correct_length_output_ids)
    # generation_outputs["logprobs"] = torch.tensor(correct_length_logprobs)

    # print(new_output_ids[0])
    # print(new_generation_lengths[0])
    # print(new_unpadded_sequence_lengths[0])
    # # print(new_logprobs[0])
    # print(tokenizer.batch_decode(torch.tensor(new_output_ids[0])))
    # print(input_lengths, new_generation_lengths, new_unpadded_sequence_lengths)

    print(f"original_is_french: {list(original_is_french)}\ntranslated_is_french: {list(translated_is_french)}")
    return generation_outputs_translation, generation_outputs, list(original_is_french) + list(translated_is_french)