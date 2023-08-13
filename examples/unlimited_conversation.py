from agentware.agent import Agent
from agentware.base import PromptProcessor
from agentware import hub

prompt_processor = PromptProcessor(
    "Forget about everythinig you were told before. You are a servant of a family, you know everything about the house and helps the family with housework. When asked a question, you can always answer it with your knowledge. When getting an instruction, try your best to use any tool to complete it. When being told a statement, answer with gotcha or some simple comment", "")
agent_id = "Alice"
if hub.agent_exists(agent_id):
    hub.remove_agent(agent_id)
hub.register_agent(agent_id)

agent = Agent(agent_id, prompt_processor)
with agent.update():
    print("AI response:", agent.run("Hi, I'm Joe"))
    print("AI response", agent.run(
        "Mom bought a fish just now. It's on the second layer of the fridge"))
    print("agent memory is", agent._memory.get_memory())
    print("AI response", agent.run("Where is the fish?"))

with agent.update():
    print("AI response:", agent.run("Ok, I moved it to a plate on the table"))

exit()
agent = Agent.pull(agent_id)
print("AI response:", agent.run("Where's the fish?"))


# 问题:
# 1. 做reflection的时候，没有带上domain knowledge. 实际上是要带的，否则有些指代不明，比如"I moved it to the plate". 另外，reflection也做得不好，比如总结出来的facts不能自相矛盾。目前是有自相矛盾的，比如fish在不同地方会同时成为facts
# 2.


# For each pair below, output if they are contradictory or not
# 1.
