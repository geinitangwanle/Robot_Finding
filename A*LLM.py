import openai
openai.api_base = "https://api.chatanywhere.tech/v1"
openai.api_key = "key"

from llmastar.pather import AStar, LLMAStar
query0 = {"start": [20, 15], "goal": [27, 15], "size": [51, 31],
        "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
        "vertical_barriers": [[25, 10, 22]],
        "range_x": [0, 51], "range_y": [0, 31]}
astar = AStar().searching(query=query0, filepath='astar.png')
llm = LLMAStar(llm='gpt', prompt='standard').searching(query=query0, filepath='gpt.png')