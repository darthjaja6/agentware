# Agentware

Agentware is an AI agent library. The agent builds knowledge base on the fly when doing daily job. Agentware has a client and a server. The client is the agentware library, which handles conversation, LLM execution, memory management, etc. The server is a combination of vector database and key-value database, which stores the knowledge base and historical memory of the agent.

## Main Features

- On the fly learning: During conversation with the user, the agent reflects and extracts knowledge. The knowledge can then be used any time the user comes back to relevant topic.
- Unlimited conversation: The agent compresses memory dynamically with reflection, so that memory length is controlled within a limit without losing context

## Quick start guide

First, `cd /agentware/agentware_server` and then run the server with docker use

```
docker-compose up
```

You'll see tons of logs.
To verify the server is launched, simply `curl http://localhost:8741/ping` and you will get a `pong` if things work fine.

## Examples

### On the fly learning

In `examples/space_travellers.py`, a housework robot is chatting with a family member. You can simply run `examples/space_travellers.py`, but in order to get a better sense of how on the fly learning is done, follow the steps here.
First, setup and register agent.

```
from agentware.agent import Agent
from agentware.base import PromptProcessor
from agentware.agent_logger import Logger

logger = Logger()
logger.set_level(Logger.INFO)

prompt_processor = PromptProcessor(
    "Forget about everythinig you were told before. You are a servant of a family, you know everything about the house and helps the family with housework. When asked a question, you can always answer it with your knowledge. When getting an instruction, try your best to use any tool to complete it. When being told a statement, answer with gotcha or some simple comment", "")
agent_id = "Alice"

agent = Agent(agent_id, prompt_processor)
agent.register(override=True)
```

A few notes:

1. logging level is set to INFO to avoid tons of debug output. If you want to see what's going on underneath, set it to `Logger.DEBUG`` or simply get rid of all the logger codes here.
2. agent is registered after creation, this is necessary so that the backend knows where to store the knowledge base, and where to fetch knowledge if you use the same agent next time

Then, talk to the agent

```
with agent.update():
    print("AI response:", agent.run("Hi, I'm Joe"))
    print("AI response", agent.run(
        "Mom bought a fish just now. It's on the second layer of the fridge"))
    print("agent memory is", agent._memory.get_memory())
    print("AI response", agent.run("Where is the fish?"))
```

`with agent.update()` tells the agent all information inside are trustworthy so its knowledgebase can be updated accordingly. Make sure you use it if you want the agent to learn from the conversation.

After this, you can simply stop the program or chat with the agent on some other topic. What's going on underneath is that the old working memory graduatelly fades away and eventually gets cleared. We mimic this situation by creating a whole new agent by pulling with the agent id.

### Unlimited conversation

In `examples/space_travellers.py`, two space travellers reunit and chat about each others' experience in travelling through the galaxy. Simply `cd examples` and run it with `python3 space_travellers.py`, the conversation conversation can continue forever(Watch out for your OpenAI api balance!). You can also view the knowledge about the planets, species, etc. of their world in the knowledge base.

```
agent = Agent.pull(agent_id)
with agent.update():
    print("AI response:", agent.run(
        "Ok, I moved the fish to a plate on the table"))
```

Now from this conversation, the agent learns what happens to the fish(it's location is changed), and updates the knowledge base to reflect this change.

We can again go away to do something else. Then when we come back and ask the fish's location, the agent will tell us the most updated information.

```
agent = Agent.pull(agent_id)
print("AI response:", agent.run("Where's the fish?"))
```

## Quick start

```
from agentware.agent import Agent

cfg = {
    "name": "Nancy",
    "description": "You are a personal assistant aimed at helping humans manage daily errands"
}
agent = Agent.init(cfg)
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
```

## Release notes

1. Aug 3, 2023

- Unlimited conversation based on memory compression and
-
