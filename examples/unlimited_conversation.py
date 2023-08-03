from agentware.agent import Agent

cfg = {
    "name": "Alice",
    "description": "You are a personal assistant aimed at helping humans manage daily errands"
}
agent = Agent.init(cfg)
if agent.exists():
    agent.remove()
agent.register()
with agent.update():
    print("AI response:", agent.run("Hi, I'm Joe"))
    print("AI response", agent.run(
        "Mom bought a fish just now. It's on the second layer of the fridge"))

agent = Agent.pull(cfg["name"])
print("AI response", agent.run("Where's the fish?"))

with agent.update():
    print("AI response:", "Ok, I moved it to a plate on the table")

agent = Agent.pull(cfg["name"])
print("AI response:", agent.run("Where's the fish?"))
