from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import gradio as gr

# Set gemini pro as LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose=True,
                             temperature=0.5,
                             google_api_key="")

duckduckgo_search = DuckDuckGoSearchRun()

def create_healthcare_workflow(age, gender, disease):
    # Define Agents
    patient_data_collector = Agent(
        role="Patient Data Collector",
        goal="Gather patient data from wearables, EHRs, and other sources.",
        backstory="Specializes in data collection from various health-related sources.",
        verbose=True,
        llm=llm,
        tools=[duckduckgo_search],  # Assumes integration with necessary data collection tools
        allow_delegation=True,
    )
    
    health_monitor = Agent(
        role="Health Monitor",
        goal="Analyze patient data to identify trends, potential issues, and medication adherence.",
        backstory="Expert in analyzing patient data for health monitoring.",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )
    
    patient_assistant = Agent(
        role="Patient Assistant",
        goal="Provide medication reminders, educational resources, and answer basic patient questions.",
        backstory="Functions as a patient support agent, offering reminders and educational information.",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )
    
    care_coordinator = Agent(
        role="Care Coordinator",
        goal="Prioritize cases based on health monitor outputs and initiate communication with healthcare professionals.",
        backstory="Manages and coordinates care based on data analysis results.",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )
    
    healthcare_professional_assistant = Agent(
        role="Healthcare Professional Assistant",
        goal="Summarize patient data, highlight concerns, and suggest potential actions.",
        backstory="Assists healthcare professionals by summarizing patient data and providing actionable insights.",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )

    # Define Tasks
    task1 = Task(
        description="Collect patient data from various sources.",
        agent=patient_data_collector,
        llm=llm
    )
    
    task2 = Task(
        description="Analyze collected patient data to identify trends and issues.",
        agent=health_monitor,
        llm=llm
    )
    
    task3 = Task(
        description="Provide medication reminders and educational resources.",
        agent=patient_assistant,
        llm=llm
    )
    
    task4 = Task(
        description="Prioritize cases and communicate with healthcare professionals based on data analysis.",
        agent=care_coordinator,
        llm=llm
    )
    
    task5 = Task(
        description="Summarize patient data and suggest potential actions for healthcare professionals.",
        agent=healthcare_professional_assistant,
        llm=llm
    )

    # Create Crew
    health_crew = Crew(
        agents=[patient_data_collector, health_monitor, patient_assistant, care_coordinator, healthcare_professional_assistant],
        tasks=[task1, task2, task3, task4, task5],
        verbose=2,
        process=Process.sequential,
    )

    # Run the Crew
    crew_result = health_crew.kickoff()

    return crew_result

# Gradio interface
def run_healthcare_app(age, gender, disease):
    crew_result = create_healthcare_workflow(age, gender, disease)
    return crew_result

iface = gr.Interface(
    fn=run_healthcare_app, 
    inputs=["text", "text", "text"], 
    outputs="text",
    title="Healthcare Workflow Management",
    description="Enter age, gender, and disease status to receive personalized healthcare workflow management."
)

iface.launch()
