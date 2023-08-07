from agentware.agent import Agent
from agentware import hub

cfg = {
    "name": "Alice",
    "description": "You are a personal assistant aimed at helping humans manage daily errands"
}
agent_id = cfg["name"]
if hub.agent_exists(agent_id):
    hub.remove_agent(agent_id)
hub.register_agent(agent_id)

agent = Agent.init(cfg)
with agent.update():
    print("AI response:", agent.run("Hi, I'm Joe"))
    print("AI response", agent.run(
        "Mom bought a fish just now. It's on the second layer of the fridge"))
    print("AI response", agent.run("Where's the fish?"))

with agent.update():
    print("AI response:", agent.run("Ok, I moved it to a plate on the table"))

agent = Agent.pull(agent_id)
print("AI response:", agent.run("Where's the fish?"))


# 问题:
# 1. 做reflection的时候，没有带上domain knowledge. 实际上是要带的，否则有些指代不明，比如"I moved it to the plate". 另外，reflection也做得不好，比如总结出来的facts不能自相矛盾。目前是有自相矛盾的，比如fish在不同地方会同时成为facts
# 2.


# For each pair below, output if they are contradictory or not
# 1.
