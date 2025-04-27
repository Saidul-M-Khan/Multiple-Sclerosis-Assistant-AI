import os
import json
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from openai import OpenAI
from dotenv import load_dotenv
from .rag_system import query_knowledge_base
from datetime import datetime


load_dotenv()

# OpenAI Client Setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load MS symptoms knowledge base
MS_SYMPTOMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/ms_symptoms.json")

def load_ms_symptoms():
    try:
        with open(MS_SYMPTOMS_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create default symptoms data if file doesn't exist
        default_symptoms = {
            "common_symptoms": [
                {"name": "Fatigue", "description": "Extreme tiredness due to MS-related nerve damage and energy depletion."},
                {"name": "Vision problems", "description": "Blurred vision, double vision, or loss of vision caused by optic nerve inflammation."},
                {"name": "Numbness or tingling", "description": "Pins-and-needles sensations or loss of feeling in the face, arms, legs, or body."},
                {"name": "Muscle weakness", "description": "Reduced strength and difficulty moving due to impaired nerve conduction."},
                {"name": "Balance problems", "description": "Dizziness, vertigo, loss of coordination, and frequent falls."},
                {"name": "Cognitive changes", "description": "Memory loss, slow information processing, poor judgment, and difficulty concentrating."},
                {"name": "Pain", "description": "Neuropathic or musculoskeletal pain due to nerve or muscle dysfunction."},
                {"name": "Mobility issues", "description": "Difficulty walking, foot drop, gait disturbances, and spasticity affecting mobility."},
                {"name": "Bladder dysfunction", "description": "Urgency, frequency, incontinence, or urinary retention caused by disrupted nerve signals."},
                {"name": "Bowel problems", "description": "Constipation, bowel incontinence, or difficulty with bowel movements."}
            ],
            "less_common_symptoms": [
                {"name": "Speech difficulties", "description": "Slurred, slowed, or nasal-sounding speech due to impaired motor control."},
                {"name": "Swallowing problems", "description": "Difficulty swallowing or choking due to muscle weakness (dysphagia)."},
                {"name": "Sexual dysfunction", "description": "Reduced libido, erectile dysfunction, or anorgasmia caused by nerve damage."},
                {"name": "Tremor", "description": "Involuntary shaking or trembling due to cerebellar or motor nerve damage."},
                {"name": "Seizures", "description": "Rare but possible episodes of uncontrolled electrical brain activity."},
                {"name": "Breathing problems", "description": "Shortness of breath, irregular breathing, or reduced lung function from muscle weakness."},
                {"name": "Hearing loss", "description": "Rare auditory impairment from damage to the auditory nerves or brain pathways."},
                {"name": "Mood changes", "description": "Depression, anxiety, mood swings, emotional instability, or pseudobulbar affect."},
                {"name": "Burning or itching sensations", "description": "False burning or itching feelings caused by nerve dysfunction."},
                {"name": "Sensory overload", "description": "Heightened sensitivity to sensory stimuli such as light, sound, or touch."},
                {"name": "Tight band-like sensation (MS Hug)", "description": "Feeling of tightness or pressure around the torso due to muscle spasms."},
                {"name": "Spasticity", "description": "Muscle stiffness and resistance to movement from nerve damage."},
                {"name": "Muscle spasms and cramps", "description": "Painful involuntary muscle contractions."},
                {"name": "Clumsiness", "description": "Frequent tripping or dropping objects due to motor coordination problems."},
                {"name": "Restless legs syndrome", "description": "Urge to move legs with uncomfortable sensations, worse at night."},
                {"name": "Heat sensitivity", "description": "Temporary worsening of symptoms in hot environments (Uhthoff's phenomenon)."},
                {"name": "Cold sensitivity", "description": "Increased discomfort or stiffness in cold temperatures."},
                {"name": "Posture problems", "description": "Poor posture and back pain due to muscle weakness or imbalance."},
                {"name": "Deconditioning", "description": "Muscle wasting and fatigue from reduced activity."},
                {"name": "Sleep disturbances", "description": "Difficulty sleeping, insomnia, or fragmented sleep."},
                {"name": "Headaches", "description": "Migraine-like headaches triggered by MS-related changes."},
                {"name": "Trigeminal neuralgia", "description": "Sharp, stabbing facial pain due to trigeminal nerve irritation."},
                {"name": "Facial weakness", "description": "Drooping or difficulty moving parts of the face."},
                {"name": "Hiccups", "description": "Persistent hiccups from diaphragm spasms caused by nerve dysfunction."},
                {"name": "Hearing problems", "description": "Tinnitus or hearing loss caused by auditory nerve damage."}
            ],
            "symptom_patterns": [
                {"name": "Relapsing-remitting", "description": "Periods of new or worsening symptoms followed by recovery."},
                {"name": "Secondary progressive", "description": "Gradual worsening of symptoms after an initial relapsing phase."},
                {"name": "Primary progressive", "description": "Steady progression of disability without early relapses or remissions."},
                {"name": "Progressive-relapsing", "description": "Continuous disease progression with occasional acute relapses."}
            ]
        }
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(MS_SYMPTOMS_PATH), exist_ok=True)
        
        # Save default symptoms
        with open(MS_SYMPTOMS_PATH, "w") as f:
            json.dump(default_symptoms, f, indent=2)
            
        return default_symptoms

# Initialize MS symptoms knowledge base
ms_symptoms = load_ms_symptoms()


class MSHealthAI:
    """Handler for MS-specialized AI model integration"""
    
    @staticmethod
    def generate_title(question: str) -> Optional[str]:
        """Generate a title for a chat session based on the first question"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """Based on the Multiple Sclerosis (MS) related query below, generate a concise, 
                        empathetic title that does not exceed 30 characters. The title should be 
                        descriptive but maintain privacy, avoiding overly specific details. Focus on 
                        the general theme of the conversation rather than specific symptoms."""
                    },
                    {
                        "role": "user", 
                        "content": question
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating title: {e}")
            # Provide a default title if generation fails
            return "MS Support Conversation"
    
    @staticmethod
    def get_conversation_chain(user_email: str = None):
        """Create an MS-focused conversational chain using the OpenAI API and RAG with FAISS"""
        def process_query(question: str, history: List[Dict] = None):
            try:
                # Get relevant context from knowledge base if user_email is provided
                context_str = ""
                if user_email:
                    context_results = query_knowledge_base(user_email, question, top_k=3)
                    if context_results:
                        context_items = []
                        for idx, item in enumerate(context_results):
                            content_snippet = item['content'][:500] + "..." if len(item['content']) > 500 else item['content']
                            context_items.append(
                                f"Document {idx+1}: {item['document_title']} (Relevance: {item['score']:.2f})\n"
                                f"Content: {content_snippet}"
                            )
                        context_str = "\n\n".join(context_items)
                
                messages = []

                # System message to guide the AI
                system_prompt = f"""You are a specialized Multiple Sclerosis (MS) support assistant, trained to provide 
                compassionate, evidence-based guidance. Your role is to listen carefully, understand the person's concerns 
                related to MS, and provide supportive, helpful responses. Respond in a warm, conversational manner while 
                maintaining professionalism. VERY IMPORTANT: Maintain full memory of the entire conversation history to 
                provide consistent, contextually relevant responses.

                About Multiple Sclerosis:
                - MS is a chronic autoimmune disease affecting the central nervous system
                - It causes damage to myelin, the protective covering of nerve fibers
                - Symptoms vary widely between individuals and can fluctuate over time
                - Common symptoms include fatigue, mobility issues, numbness, vision problems, 
                  cognitive changes, pain, and bladder/bowel dysfunction
                - MS has different patterns: relapsing-remitting, secondary progressive, 
                  primary progressive, and progressive-relapsing

                Remember to:
                1. Show empathy and deep understanding of MS challenges
                2. Ask clarifying questions about specific symptoms when appropriate
                3. Provide evidence-based information and management strategies specific to MS
                4. Emphasize that you're an AI and not a replacement for healthcare professionals
                5. Encourage maintaining regular contact with MS specialists and neurologists
                6. Reference previous parts of the conversation when relevant
                7. Address both physical and emotional aspects of living with MS
                8. Acknowledge the variability and unpredictability of MS symptoms

                Always prioritize the person's wellbeing and safety in your responses.
                """
                
                if context_str:
                    system_prompt += f"\n\nAdditional context from relevant documents (use this information to enhance your response):\n{context_str}"
                
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

                # Add ALL conversation history if available
                if history:
                    for item in history:
                        messages.append({"role": "user", "content": item["query_text"]})
                        messages.append({"role": "assistant", "content": item["response_text"]})

                # Add the current question
                messages.append({"role": "user", "content": question})

                # Get response from OpenAI with increased token limit
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1000,  # Increase token limit for longer responses
                    temperature=0.7  # Slightly increased creativity for empathetic responses
                )

                return {"answer": response.choices[0].message.content}
            except Exception as e:
                # If context is too long, try with truncated history
                if "maximum context length" in str(e).lower():
                    # Keep only the most recent 10 exchanges if history is too long
                    truncated_history = history[-10:] if history and len(history) > 10 else history
                    messages = [
                        {
                            "role": "system",
                            "content": """You are a specialized Multiple Sclerosis (MS) support assistant.
                            Due to technical limitations, only the most recent portion of the conversation history is available,
                            but please maintain continuity as best as possible. [Same MS guidelines as above]"""
                        }
                    ]

                    if truncated_history:
                        for item in truncated_history:
                            messages.append({"role": "user", "content": item["query_text"]})
                            messages.append({"role": "assistant", "content": item["response_text"]})

                    messages.append({"role": "user", "content": question})

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages
                    )
                    return {"answer": response.choices[0].message.content}
                else:
                    raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

        return {"invoke": process_query}

    @staticmethod
    async def upload_training_document(file_path: str, user_email: str = None, document_title: str = None, document_description: str = None):
        """Process and upload a document for RAG training"""
        try:
            # Extract file extension
            file_extension = os.path.splitext(file_path)[1][1:].lower()
            
            # Generate a collection name based on timestamp and user
            collection_name = f"{user_email.split('@')[0] if user_email else 'anonymous'}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Import process_file from rag_system
            from .rag_system import process_file
            
            # Process the file
            result = await process_file(
                file_path=file_path,
                file_type=file_extension,
                user_email=user_email or "anonymous",
                collection_name=collection_name,
                document_title=document_title or os.path.basename(file_path),
                document_description=document_description
            )
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    @staticmethod
    def analyze_symptoms(symptom_text: str):
        """Analyze MS symptoms and provide tailored recommendations"""
        try:
            # Load current MS symptoms knowledge
            ms_symptom_data = load_ms_symptoms()
            
            # Format symptoms data for the prompt
            symptoms_info = []
            for category, symptoms in ms_symptom_data.items():
                symptoms_info.append(f"## {category.replace('_', ' ').title()}")
                for symptom in symptoms:
                    symptoms_info.append(f"- {symptom['name']}: {symptom['description']}")
            
            symptoms_context = "\n".join(symptoms_info)
            
            system_prompt = f"""
            You are a compassionate MS support assistant specializing in symptom assessment. Your role is to:

            1. Carefully analyze the person's described MS-related symptoms
            2. Match their experiences with known MS symptoms
            3. Identify which symptoms align with MS and which might need different attention
            4. Provide thoughtful, evidence-based recommendations specific to MS management
            5. Suggest coping strategies and self-care practices tailored to MS

            Here is the current knowledge base of MS symptoms to reference:
            
            {symptoms_context}
            
            Structure your response in a warm, conversational manner that includes:

            1. A gentle acknowledgment of their feelings and concerns
            2. Identification of which described experiences match known MS symptoms
            3. A thoughtful analysis of what might be happening (avoiding definitive diagnosis)
            4. Practical, actionable suggestions for MS symptom management
            5. Clear guidance on which symptoms warrant contacting their neurologist promptly
            6. Encouragement to maintain regular contact with their MS care team
            7. A compassionate closing note

            Remember that this is supportive information based on limited text, not a clinical diagnosis. 
            Be supportive, empathetic, and helpful while maintaining appropriate boundaries.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"I've been experiencing these symptoms with my MS: '{symptom_text}'"}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return {
                "analysis": response.choices[0].message.content.strip()
            }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing MS symptoms: {str(e)}")
        
    @staticmethod
    def update_symptoms_knowledge(new_symptoms_data: Dict[str, List[Dict[str, str]]]):
        """Update the MS symptoms knowledge base with new information"""
        try:
            # Validate the structure of the input data
            required_keys = ["common_symptoms", "less_common_symptoms", "symptom_patterns"]
            for key in required_keys:
                if key not in new_symptoms_data:
                    raise ValueError(f"Missing required key in symptoms data: {key}")
                
                if not isinstance(new_symptoms_data[key], list):
                    raise ValueError(f"The value for '{key}' must be a list")
                
                for item in new_symptoms_data[key]:
                    if not isinstance(item, dict) or "name" not in item or "description" not in item:
                        raise ValueError(f"Each symptom in '{key}' must have 'name' and 'description' fields")
            
            # Save the updated symptoms data
            with open(MS_SYMPTOMS_PATH, "w") as f:
                json.dump(new_symptoms_data, f, indent=2)
            
            # Update the global variable
            global ms_symptoms
            ms_symptoms = new_symptoms_data
            
            return {"status": "success", "message": "MS symptoms knowledge base updated successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating MS symptoms knowledge: {str(e)}")