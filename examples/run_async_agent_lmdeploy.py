import asyncio
import json
import time

from datasets import load_dataset

from fire.agents.stream import PLUGIN_CN, AsyncAgentForfire, AsyncMathCoder, get_plugin_prompt
from fire.llms import fire2_META
from fire.llms.lmdeploy_wrapper import AsyncLMDeployPipeline
from fire.prompts.parsers import PluginParser

# set up the loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# initialize the model
model = AsyncLMDeployPipeline(
    path='fire/fire2_5-7b-chat',
    meta_template=fire2_META,
    model_name='fire-chat',
    tp=1,
    top_k=1,
    temperature=1.0,
    stop_words=['<|im_end|>', '<|action_end|>'],
    max_new_tokens=1024,
)

# ----------------------- interpreter -----------------------
print('-' * 80, 'interpreter', '-' * 80)

ds = load_dataset('lighteval/MATH', split='test')
problems = [item['problem'] for item in ds.select(range(0, 5000, 2))]

coder = AsyncMathCoder(
    llm=model,
    interpreter=dict(
        type='fire.actions.AsyncIPythonInterpreter', max_kernels=300),
    max_turn=11)
tic = time.time()
coros = [coder(query, session_id=i) for i, query in enumerate(problems)]
res = loop.run_until_complete(asyncio.gather(*coros))
# print([r.model_dump_json() for r in res])
print('-' * 120)
print(f'time elapsed: {time.time() - tic}')

with open('./tmp_1.json', 'w') as f:
    json.dump([coder.get_steps(i) for i in range(len(res))],
              f,
              ensure_ascii=False,
              indent=4)

# ----------------------- plugin -----------------------
print('-' * 80, 'plugin', '-' * 80)
plugins = [dict(type='fire.actions.AsyncArxivSearch')]
agent = AsyncAgentForfire(
    llm=model,
    plugins=plugins,
    output_format=dict(
        type=PluginParser,
        template=PLUGIN_CN,
        prompt=get_plugin_prompt(plugins)))

tic = time.time()
coros = [
    agent(query, session_id=i)
    for i, query in enumerate(['LLM智能体方向的最新论文有哪些？'] * 50)
]
res = loop.run_until_complete(asyncio.gather(*coros))
# print([r.model_dump_json() for r in res])
print('-' * 120)
print(f'time elapsed: {time.time() - tic}')
