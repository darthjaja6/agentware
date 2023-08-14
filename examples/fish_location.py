from agentware.agent import Agent
from agentware.base import PromptProcessor

prompt_processor = PromptProcessor(
    "Forget about everythinig you were told before. You are a servant of a family, you know everything about the house and helps the family with housework. When asked a question, you can always answer it with your knowledge. When getting an instruction, try your best to use any tool to complete it. When being told a statement, answer with gotcha or some simple comment", "")
agent_id = "Alice"

agent = Agent(agent_id, prompt_processor)
agent.register(override=True)
with agent.update():
    print("AI response:", agent.run("Hi, I'm Joe"))
    print("AI response", agent.run(
        "Mom bought a fish just now. It's on the second layer of the fridge"))
    print("agent memory is", agent._memory.get_memory())
    print("AI response", agent.run("Where is the fish?"))

with agent.update():
    print("AI response:", agent.run("Ok, I moved it to a plate on the table"))

agent = Agent.pull(agent_id)
print("AI response:", agent.run("Where's the fish?"))
