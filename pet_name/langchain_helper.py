from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, AgentType, initialize_agent
from dotenv import load_dotenv

load_dotenv()

def generate_pet_names(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet and i want a cool name for it. It is {pet_color} Suggest me five cool names for my pet."
    )
    name = LLMChain(llm = llm, prompt=prompt_template_name)

    response = name({'animal_type': animal_type, 'pet_color': pet_color})
    return response['text']

def lang_agent():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(["wikipedia", 'llm-math'], llm = llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    result = agent.run("What is the average age of a dog? Multiply it by 3 and add 5.")
    print(result)


if __name__ == "__main__":
    lang_agent()