import os
from dotenv import load_dotenv
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai import Agent, Tool
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Form, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from twilio.twiml.messaging_response import MessagingResponse

# Assuming you have a logging setup (replace with your actual logging)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

# SQLite database configuration
DB_PATH = os.getenv('DB_PATH', 'patients.db')

def get_db_connection():
    """Create and return a SQLite database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None
    
def init_database():
    """Initialize the database with patient table if it doesn't exist"""
    conn = get_db_connection()
    if conn:
        try:
            with conn:
                # Patient table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS patient (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age INTEGER NOT NULL,
                        gender TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for faster case-insensitive searches
                conn.execute("CREATE INDEX IF NOT EXISTS idx_patient_name_lower ON patient(LOWER(name))")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_patient_age ON patient(age)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_patient_gender ON patient(gender)")
                
        except Exception as e:
            print(f"Database initialization error: {e}")
        finally:
            conn.close()

# Initialize database on startup
init_database()    

@dataclass
class Patient:
    name: str
    age: int
    gender: str

class PatientCreate(BaseModel):
    name: str = Field(..., description="Full name of the patient")
    age: int = Field(..., ge=0, le=150, description="Age of the patient (0-150)")
    gender: str = Field(..., description="Gender of the patient (Male/Female/Other)")    

def validate_patient_data(patient_data: dict) -> Optional[PatientCreate]:
    """Validate patient data using Pydantic model"""
    try:
        return PatientCreate(**patient_data)
    except Exception as e:
        print(f"Validation error: {e}")
        return None

@Tool
def insert_patient_validated(name: str, age: int, gender: str) -> str:
    """
    Insert a new patient into the database after validation.
    Requires all parameters: name (string), age (integer 0-150), gender (string)
    """
    # Validate the input data
    patient_data = {"name": name, "age": age, "gender": gender}
    validated_patient = validate_patient_data(patient_data)
    
    if not validated_patient:
        return "Failed to insert patient: Invalid data provided. Please provide name (string), age (integer 0-150), and gender (string)."
    
    # Proceed with database insertion
    conn = get_db_connection()
    if conn:
        try:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO patient (name, age, gender) VALUES (?, ?, ?)",
                    (validated_patient.name, validated_patient.age, validated_patient.gender)
                )
                patient_id = cursor.lastrowid
                return f"Successfully inserted patient {validated_patient.name} with ID {patient_id}"
        except Exception as e:
            print(f"Error inserting patient: {e}")
            return f"Database error: {str(e)}"
        finally:
            conn.close()
    return "Failed to insert patient: Database connection error"

@Tool
def update_patient(name: str, age: int, gender: str) -> str:
    """Update an existing patient in the database"""
    # Validate the input data first
    patient_data = {"name": name, "age": age, "gender": gender}
    validated_patient = validate_patient_data(patient_data)
    
    if not validated_patient:
        return "Failed to update patient: Invalid data provided. Please provide name (string), age (integer 0-150), and gender (string)."
    
    conn = get_db_connection()
    if conn:
        try:
            with conn:
                cursor = conn.execute(
                    "UPDATE patient SET age = ?, gender = ?, updated_at = CURRENT_TIMESTAMP WHERE LOWER(name) = LOWER(?)",
                    (validated_patient.age, validated_patient.gender, validated_patient.name)
                )
                rows_affected = cursor.rowcount
                if rows_affected > 0:
                    return f'Updated patient {validated_patient.name} successfully'
                else:
                    return f'Patient {validated_patient.name} not found for update'
        except Exception as e:
            print(f"Error updating patient: {e}")
            return f"Database error: {str(e)}"
        finally:
            conn.close()
    return "Failed to update patient: Database connection error"

@Tool
def delete_patient(name: str) -> str:
    """Delete a patient from the database"""
    conn = get_db_connection()
    if conn:
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM patient WHERE LOWER(name) = LOWER(?)",
                    (name,)
                )
                rows_affected = cursor.rowcount
                if rows_affected > 0:
                    return f'Deleted patient {name} successfully'
                else:
                    return f'Patient {name} not found for deletion'
        except Exception as e:
            print(f"Error deleting patient: {e}")
            return f"Database error: {str(e)}"
        finally:
            conn.close()
    return "Failed to delete patient: Database connection error"

@Tool
def get_patient(name: str) -> str:
    """Retrieve a patient from the database by name"""
    conn = get_db_connection()
    if conn:
        try:
            with conn:
                cursor = conn.execute(
                    "SELECT name, age, gender FROM patient WHERE LOWER(name) = LOWER(?)",
                    (name,)
                )
                result = cursor.fetchone()
                if result:
                    return f"Patient found: Name: {result['name']}, Age: {result['age']}, Gender: {result['gender']}"
                else:
                    return f'Patient {name} not found'
        except Exception as e:
            print(f"Error retrieving patient: {e}")
            return f"Database error: {str(e)}"
        finally:
            conn.close()
    return "Failed to retrieve patient: Database connection error"

@Tool
def list_all_patients() -> str:
    """List all patients in the database"""
    conn = get_db_connection()
    if conn:
        try:
            with conn:
                cursor = conn.execute("SELECT name, age, gender FROM patient ORDER BY name")
                patients = cursor.fetchall()
                if patients:
                    result = "All patients:\n"
                    for patient in patients:
                        result += f"Name: {patient['name']}, Age: {patient['age']}, Gender: {patient['gender']}\n"
                    return result
                else:
                    return "No patients found in the database"
        except Exception as e:
            print(f"Error listing patients: {e}")
            return f"Database error: {str(e)}"
        finally:
            conn.close()
    return "Failed to list patients: Database connection error"

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
provider = GoogleProvider(api_key=GEMINI_API_KEY)
model = GoogleModel('gemini-2.0-flash', provider=provider)


app = FastAPI(title="Patient Management API")

def create_agent():
    """Create and return the Agent instance."""
    return Agent(model=model, instrument=True, 
                  tools=[insert_patient_validated, update_patient, delete_patient, get_patient, list_all_patients],
                  system_prompt="You are a medical assistant. Always validate patient data thoroughly before performing " \
                  "any operations. Ensure all required fields (name, age, gender) are provided and valid before " \
                  "inserting into the database. You can help with inserting, updating, deleting, and retrieving patient records.")

# def create_agent():
#     """Create and return the Agent instance."""
#     return Agent(model=model, instrument=True,
#                   tools=[insert_patient_validated, update_patient, delete_patient, get_patient, list_all_patients],
#                   system_prompt="You are a medical assistant. Always validate patient data thoroughly before performing " \
#                   "any operations. Ensure all required fields (name, age, gender) are provided and valid before " \
#                   "inserting into the database. You can help with inserting, updating, deleting, and retrieving patient records.")


# async def process_command(command: str):
#     """Process the user command with the AI agent."""
#     message_history = []
#     result = await agent.run(command, message_history=message_history)
#     return result.output

# class Command(BaseModel):
#     command: str

# @app.post("/process_command")
# async def process_command_endpoint(command_data: Command):
#     """Endpoint to process commands."""
#     try:
#         result = await process_command(command_data.command)
#         return JSONResponse({"result": result})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Twilio integration would typically involve handling incoming message webhooks.
# This is a placeholder for where that logic would go.  You'd need to install the Twilio Python library.
# from twilio.twiml.messaging_response import MessagingResponse
#
# @app.post("/twilio_webhook")
# async def twilio_webhook(request: Request):
#     """Twilio webhook endpoint."""
#     form_data = await request.form()
#     message_body = form_data['Body']
#     # Process the message_body with the AI agent
#     result = await process_command(message_body)
#
#     # Create a TwiML response
#     twiml = MessagingResponse()
#     twiml.message(result)
#
#     return Response(content=str(twiml), media_type="application/xml")

@app.post("/chat")
async def chat_endpoint(
    From: str = Form(...),
    Body: str = Form(...),
    agent: Agent = Depends(create_agent)
):
    try:
        result = await agent.run(Body)
        twiml = MessagingResponse()
        twiml.message(result.output)
        twiml_string = str(twiml)
        headers = {'Content-Type': 'application/xml'}
        logger.info(f"TwiML Response: {twiml_string}")
        response= Response(content=twiml_string, media_type="application/xml",headers=headers)
        logger.info(f"Response Headers: {response.headers}") # Log the headers
        return response
    except Exception as e:
        twiml = MessagingResponse()
        twiml.message(f"Error: {e}")
        twiml_string = str(twiml)
        headers = {'Content-Type': 'application/xml'}
        logger.error(f"TwiML Error: {twiml_string} - Exception: {e}")
        return Response(content=twiml_string, media_type="application/xml",headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))