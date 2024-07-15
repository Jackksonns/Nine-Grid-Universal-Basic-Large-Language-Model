# pip install vllm
# vllm 0.22版本适合低nvidia驱动 如470.xx.xx,cuda11
# vllm 0.41+ 版本适合高nvidia驱动，如535.161.xx, cuda12

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = ["请介绍下启元实验室",]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.3, top_p=0.8)

# Create an LLM.
#llm = LLM(model="../models/facebook/opt-125m")
llm = LLM(model="fm9g-selfrec/", trust_remote_code=True)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
