from pymongo import MongoClient
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from typing import TypedDict, Optional
import re

# ----- Step 1: Define State Type -----
class ChatState(TypedDict):
    emp_code: str
    question: str
    employee: Optional[dict]
    response: Optional[str]

# ----- Step 2: MongoDB Helper Functions -----
def get_employee_model(emp_code):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["EMP"]
    collection = db["emp"]
    return collection.find_one({"emp_code": emp_code})

def store_change_request(emp_code, field, old_val, new_val):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["EMP"]
    change_log = db["change_requests"]
    change_log.insert_one({
        "emp_code": emp_code,
        "field": field,
        "old_value": old_val,
        "new_value": new_val,
        "status": "pending"
    })

# ----- Step 3: LLM Setup -----
llm = OllamaLLM(model="mistral")

prompt = PromptTemplate.from_template(
    """
    You are an employee data change detection system. Analyze the following:

    Employee Data:
    - Name: {name}
    - Email: {email}
    - Department: {department}
    - Shift: {shift}

    User Request: "{question}"

    If the user is requesting a change to any field, respond EXACTLY in this format:
    CHANGE|field_name|new_value

    If no change is requested, respond EXACTLY with:
    NO_CHANGE

    Examples:
    - For email change: CHANGE|email|new.email@example.com
    - For department change: CHANGE|department|IT
    - For no change: NO_CHANGE
    """
)

def extract_intent_and_value(employee, question):
    # Format the prompt with employee data and question
    formatted_prompt = prompt.format(**employee, question=question)
    # Invoke the LLM with the formatted prompt string
    response = llm.invoke(formatted_prompt)
    return response.strip()

# ----- Step 4: LangGraph Nodes -----
def get_employee(state: ChatState) -> ChatState:
    emp = get_employee_model(state["emp_code"])
    return {**state, "employee": emp}

def detect_and_store_change(state: ChatState) -> ChatState:
    emp = state["employee"]
    question = state["question"]

    if not emp:
        state["response"] = "❌ Employee not found."
        return state

    result = extract_intent_and_value(emp, question)

    if result.startswith("CHANGE|"):
        try:
            # Parse the structured response
            parts = result.split("|")
            if len(parts) == 3:
                field = parts[1].strip()
                new_value = parts[2].strip()
                old_value = emp.get(field)
                
                print(f"Detected change: {field} from {old_value} to {new_value}")  

                if new_value != old_value:
                    store_change_request(emp["emp_code"], field, old_value, new_value)
                    state["response"] = f"✅ Change request logged: {field} ➝ {new_value}"
                else:
                    state["response"] = "ℹ️ Requested value is the same as existing. No change logged."
            else:
                state["response"] = "⚠️ Invalid response format from LLM"
        except Exception as e:
            state["response"] = f"⚠️ Error processing change request: {str(e)}"
    elif result == "NO_CHANGE":
        state["response"] = "🛑 No change detected."
    else:
        state["response"] = "⚠️ Unexpected response format from LLM"

    return state

# ----- Step 5: LangGraph Setup -----
builder = StateGraph(ChatState)

builder.add_node("fetch", get_employee)
builder.add_node("process", detect_and_store_change)

builder.set_entry_point("fetch")
builder.add_edge("fetch", "process")
builder.set_finish_point("process")

chatbot_graph = builder.compile()

# ----- Step 6: Run Example -----
if __name__ == "__main__":
    result = chatbot_graph.invoke({
        "emp_code": "EMP001",  # Make sure this emp_code exists in your MongoDB collection
        "question": "could you please update my email id to rishalmundekkat@gmail.com"
    })  
    print("🤖 Response:", result["response"])
