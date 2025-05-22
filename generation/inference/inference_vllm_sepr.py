from vllm import LLM, SamplingParams
import argparse
import json

ALPACA_TEMPLATE = (
    "Nedan är en instruktion som beskriver en uppgift, parad med en ingång som ger ytterligare" 
    "sammanhang. Skriv ett svar som på lämpligt sätt kompletterar begäran.\n\n"
    "### Instruktion:\n{instruction}\n\n### Ingång :\n{input}\n\n### Svar:\n"
)

sampling_params = SamplingParams(n=5, 
                                 max_tokens=4000,
                                 repetition_penalty = 1.5)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--file_out', type=str, required=True)

    args = parser.parse_args()
    test_data = args.test_data

    # load model for sampling
    llm = LLM(args.base_model, max_num_seqs=45, tensor_parallel_size=2, max_model_len=4000) 

    with open(test_data, 'r') as f:
        test_dn = json.load(f)
    
    # create prompts in ALPACA format for sampling
    prompts = []
    for sample in test_dn:
        text = ALPACA_TEMPLATE.format(
        instruction=sample["instruction"],
        input=sample["input"])
        prompts.append(text)

    # generate responses
    outputs = llm.generate(prompts, sampling_params)

    # save 5 outputs per prompt in JSON file
    docs = []
    for output in outputs: 
        doc = {}
        doc["prompt"] = output.prompt
        doc["output1"] = output.outputs[0].text
        doc["output2"] = output.outputs[1].text
        doc["output3"] = output.outputs[2].text
        doc["output4"] = output.outputs[3].text
        doc["output5"] = output.outputs[4].text
        docs.append(doc)

    filename = args.file_out
    with open(filename, 'w') as f:
        json.dump(docs, f)

