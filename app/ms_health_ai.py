from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from pydantic import EmailStr, BaseModel
from sqlalchemy.orm import Session
from app.models import Session as DBSession, ChatMessage, User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationState(BaseModel):
    """Model for conversation state"""
    stage: str
    demographics: Dict[str, Any]
    symptoms: Dict[str, List[str]]
    diagnostic_tests: Dict[str, Dict[str, Any]]
    treatments: Dict[str, List[str]]
    lifestyle: Dict[str, List[str]]
    chat_history: List[Dict[str, str]]
    title: str
    analysis_complete: bool = False
    analysis: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for database storage"""
        return {
            "stage": self.stage,
            "demographics": self.demographics,
            "symptoms": self.symptoms,
            "diagnostic_tests": self.diagnostic_tests,
            "treatments": self.treatments,
            "lifestyle": self.lifestyle,
            "chat_history": self.chat_history,
            "title": self.title,
            "analysis_complete": self.analysis_complete,
            "analysis": self.analysis or {},
            "recommendations": self.recommendations or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create state from dictionary"""
        # Ensure all required fields are present
        required_fields = {
            "stage": "initial",
            "demographics": {},
            "symptoms": {},
            "diagnostic_tests": {},
            "treatments": {},
            "lifestyle": {},
            "chat_history": [],
            "title": "New MS Consultation",
            "analysis_complete": False,
            "analysis": {},
            "recommendations": {}
        }
        
        # Update with provided data
        state_data = {**required_fields, **data}
        return cls(**state_data)

class MSHealthAIError(Exception):
    """Base exception for MS Health AI errors"""
    pass

class ValidationError(MSHealthAIError):
    """Raised when input validation fails"""
    pass

class StateError(MSHealthAIError):
    """Raised when there's an issue with conversation state"""
    pass

class InvalidStateError(MSHealthAIError):
    """Raised when the conversation state is invalid"""
    pass

class ParsingError(MSHealthAIError):
    """Raised when there's an error parsing user input"""
    pass

class StateManager:
    """Manages conversation state persistence in database"""
    def __init__(self, db: Session):
        self.db = db

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state from database"""
        session = self.db.query(DBSession).filter(DBSession.id == session_id).first()
        if not session:
            return None
        return session.ai_state

    def update_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """Update session state in database"""
        session = self.db.query(DBSession).filter(DBSession.id == session_id).first()
        if session:
            session.ai_state = state
            self.db.commit()

class MSHealthAI:
    """Main AI class for MS health assistance"""
    def __init__(self, db: Session):
        """
        Initialize MSHealthAI with knowledge base and conversation state
        
        Args:
            db: SQLAlchemy database session
        """
        try:
            self.db = db
            self.knowledge_base: Dict[str, Any] = self._load_knowledge_base()
            self.conversation_state: Dict[str, ConversationState] = {}
            self.state_manager = StateManager(db)
        except Exception as e:
            logger.error(f"Error initializing MSHealthAI: {str(e)}")
            raise MSHealthAIError("Failed to initialize MS Health AI system")
    
    def _load_knowledge_base(self) -> Dict:
        try:
            return {
                "mycotoxin_tests": {
                    "ochratoxin_a": {
                        "name": "Ochratoxin A",
                        "reference_ranges": {
                            "not_present": "<1.8",
                            "equivocal": "1.8 to <2",
                            "present": ">=2"
                        },
                        "symptoms": ["Fatigue", "Dermatitis", "Irritated bowel"],
                        "disease_states": ["Kidney disease", "Cancer"],
                        "activity": "Inhibits mitochondrial ATP, potent teratogen, and immune suppressor",
                        "mechanism": "Disrupts cellular energy production and immune function",
                        "treatment_considerations": [
                            "Support kidney function",
                            "Enhance detoxification pathways",
                            "Address immune system support"
                        ]
                    },
                    "aflatoxin_group": {
                        "name": "Aflatoxin Group (B1, B2, G1, G2)",
                        "reference_ranges": {
                            "not_present": "<0.8",
                            "equivocal": "0.8 to <1",
                            "present": ">=1"
                        },
                        "symptoms": ["Shortness of breath", "Weight loss", "Impaired fetal growth"],
                        "disease_states": ["Liver disease", "Kidney disease", "Lung cancer"],
                        "activity": "Binds DNA and proteins, inhibits DNA and RNA replication",
                        "mechanism": "Causes DNA damage and cellular dysfunction",
                        "treatment_considerations": [
                            "Liver support",
                            "DNA repair support",
                            "Antioxidant therapy"
                        ]
                    },
                    "trichothecene_group": {
                        "name": "Trichothecene Group",
                        "reference_ranges": {
                            "not_present": "<0.07",
                            "equivocal": "0.07 to <0.09",
                            "present": ">=0.09"
                        },
                        "symptoms": ["Fatigue", "Weakened immune system", "Breathing issues"],
                        "disease_states": ["Bleeding disorders", "Nervous system disorders"],
                        "activity": "DNA, RNA, and protein synthesis inhibition",
                        "mechanism": "Disrupts cellular protein synthesis",
                        "treatment_considerations": [
                            "Immune system support",
                            "Respiratory support",
                            "Nervous system support"
                        ]
                    },
                    "gliotoxin": {
                        "name": "Gliotoxin Derivative",
                        "reference_ranges": {
                            "not_present": "<0.5",
                            "equivocal": "0.5 to <1",
                            "present": ">=1"
                        },
                        "symptoms": ["Memory issues", "Breathing issues"],
                        "disease_states": ["Immune dysfunction disorders"],
                        "activity": "Attacks intracellular function in immune system",
                        "mechanism": "Disrupts immune cell function",
                        "treatment_considerations": [
                            "Immune system modulation",
                            "Cognitive support",
                            "Respiratory support"
                        ]
                    },
                    "zearalenone": {
                        "name": "Zearalenone",
                        "reference_ranges": {
                            "not_present": "<0.5",
                            "equivocal": "0.5 to <0.7",
                            "present": ">=0.7"
                        },
                        "symptoms": ["Early puberty", "Low sperm counts"],
                        "disease_states": ["Cancer"],
                        "activity": "Estrogen mimic",
                        "mechanism": "Disrupts hormonal balance",
                        "treatment_considerations": [
                            "Hormonal balance support",
                            "Detoxification support",
                            "Reproductive health support"
                        ]
                    }
                },
                "mycotoxin_symptoms": {
                    "physical": [
                        "Headaches & Dizziness",
                        "Nosebleeds",
                        "Painful Lymph Nodes",
                        "Asthma",
                        "Shortness Of Breath",
                        "Gastrointestinal distress",
                        "Decreased Libido",
                        "Hair Loss",
                        "Brain Fog",
                        "Sinusitis & Sinus Issues",
                        "Hearing Problems",
                        "Cardiac Arrhythmias",
                        "Abdominal Pain and Discomfort",
                        "Numbness and Tingling In Hands",
                        "Uncomfortable or Frequent Urination",
                        "Rashes & Hives",
                        "Muscles and Joint Aches and Pains",
                        "Fluid Retention",
                        "Numbness and Tingling In Feet"
                    ],
                    "systemic": [
                        "Depression",
                        "Anxiety",
                        "Chronic Fatigue",
                        "Chronic Illness",
                        "General Weakness",
                        "Immune Suppression",
                        "Anemia",
                        "Night Sweats"
                    ]
                },
                "mycotoxin_interactions": {
                    "synergistic": [
                        "Ochratoxin A + Aflatoxin: Enhanced kidney toxicity",
                        "Trichothecene + Gliotoxin: Enhanced immune suppression",
                        "Zearalenone + Aflatoxin: Enhanced hormonal disruption"
                    ],
                    "antagonistic": [
                        "Gliotoxin + Zearalenone: Reduced hormonal effects",
                        "Ochratoxin A + Trichothecene: Reduced immune suppression"
                    ]
                },
                "treatment_protocols": {
                    "detoxification": [
                        "Binders: Activated charcoal, bentonite clay",
                        "Liver support: Milk thistle, dandelion root",
                        "Kidney support: Nettle leaf, dandelion leaf",
                        "Immune support: Vitamin C, zinc, selenium"
                    ],
                    "dietary": [
                        "Anti-inflammatory diet",
                        "High antioxidant foods",
                        "Cruciferous vegetables",
                        "Omega-3 fatty acids"
                    ],
                    "lifestyle": [
                        "Regular exercise",
                        "Stress management",
                        "Adequate sleep",
                        "Environmental control"
                    ]
                },
                "ms_types": {
                    "relapsing_remitting": {
                        "name": "Relapsing-Remitting MS (RRMS)",
                        "description": "Most common form of MS, characterized by clearly defined attacks followed by periods of remission",
                        "symptoms": ["Fatigue", "Numbness", "Vision problems", "Muscle weakness", "Coordination problems"]
                    },
                    "primary_progressive": {
                        "name": "Primary Progressive MS (PPMS)",
                        "description": "Steady worsening of neurological function from the onset of symptoms",
                        "symptoms": ["Walking difficulties", "Stiffness", "Balance problems", "Bladder problems"]
                    },
                    "secondary_progressive": {
                        "name": "Secondary Progressive MS (SPMS)",
                        "description": "Follows an initial relapsing-remitting course, then becomes steadily progressive",
                        "symptoms": ["Increasing disability", "Fewer relapses", "More progressive symptoms"]
                    },
                    "progressive_relapsing": {
                        "name": "Progressive-Relapsing MS (PRMS)",
                        "description": "Rare form of MS, characterized by steady progression with acute relapses",
                        "symptoms": ["Steady progression", "Acute attacks", "No remission periods"]
                    }
                },
                "symptoms": {
                    "physical": [
                        "Fatigue",
                        "Numbness or tingling",
                        "Muscle weakness",
                        "Vision problems",
                        "Balance problems",
                        "Coordination difficulties",
                        "Tremors",
                        "Spasticity",
                        "Pain",
                        "Bladder problems",
                        "Bowel problems",
                        "Sexual dysfunction",
                        "Speech problems",
                        "Swallowing difficulties",
                        "Walking difficulties"
                    ],
                    "cognitive": [
                        "Memory problems",
                        "Difficulty concentrating",
                        "Problem-solving issues",
                        "Information processing speed",
                        "Attention problems",
                        "Executive function difficulties",
                        "Visual-spatial problems"
                    ],
                    "emotional": [
                        "Depression",
                        "Anxiety",
                        "Mood swings",
                        "Irritability",
                        "Stress",
                        "Emotional lability"
                    ]
                },
                "diagnostic_tests": {
                    "mri": {
                        "name": "Magnetic Resonance Imaging (MRI)",
                        "description": "Primary imaging tool for MS diagnosis",
                        "findings": ["Lesions", "Brain atrophy", "Spinal cord lesions"]
                    },
                    "evoked_potentials": {
                        "name": "Evoked Potentials",
                        "description": "Measures electrical activity in response to stimuli",
                        "types": ["Visual", "Somatosensory", "Brainstem auditory"]
                    },
                    "spinal_tap": {
                        "name": "Spinal Tap (Lumbar Puncture)",
                        "description": "Analyzes cerebrospinal fluid",
                        "findings": ["Oligoclonal bands", "Elevated IgG index"]
                    },
                    "blood_tests": {
                        "name": "Blood Tests",
                        "description": "Rules out other conditions",
                        "types": ["Vitamin D", "B12", "Thyroid function", "Autoimmune markers"]
                    }
                },
                "treatments": {
                    "disease_modifying": [
                        "Interferon beta-1a",
                        "Interferon beta-1b",
                        "Glatiramer acetate",
                        "Fingolimod",
                        "Dimethyl fumarate",
                        "Teriflunomide",
                        "Natalizumab",
                        "Ocrelizumab",
                        "Alemtuzumab"
                    ],
                    "symptom_management": {
                        "fatigue": ["Amphetamines", "Modafinil", "Lifestyle modifications"],
                        "spasticity": ["Baclofen", "Tizanidine", "Physical therapy"],
                        "pain": ["Gabapentin", "Pregabalin", "Amitriptyline"],
                        "bladder": ["Oxybutynin", "Tolterodine", "Behavioral modifications"],
                        "depression": ["SSRIs", "SNRIs", "Psychotherapy"]
                    }
                },
                "lifestyle_factors": {
                    "diet": [
                        "Mediterranean diet",
                        "Vitamin D supplementation",
                        "Omega-3 fatty acids",
                        "Antioxidant-rich foods"
                    ],
                    "exercise": [
                        "Aerobic exercise",
                        "Strength training",
                        "Balance exercises",
                        "Flexibility training"
                    ],
                    "stress_management": [
                        "Meditation",
                        "Yoga",
                        "Mindfulness",
                        "Cognitive behavioral therapy"
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise MSHealthAIError("Failed to load knowledge base")
    
    def process_message(self, session_id: str, message: str, email: EmailStr) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            session_id: Unique identifier for the conversation session
            message: User's message text
            email: User's email address
            
        Returns:
            str: AI's response to the user
            
        Raises:
            ValidationError: If required parameters are missing or invalid
            StateError: If there's an issue with conversation state
            MSHealthAIError: For other AI-related errors
        """
        try:
            # Validate input parameters
            if not session_id or not isinstance(session_id, str):
                raise ValidationError("Invalid session ID")
            if not message or not isinstance(message, str):
                raise ValidationError("Invalid message")
            if not email or not isinstance(email, str):
                raise ValidationError("Invalid email")

            # Get or create session
            session = self.db.query(DBSession).filter(DBSession.id == session_id).first()
            if not session:
                # Create new user if needed
                user = self.db.query(User).filter(User.email == email).first()
                print(user)
                if not user:
                    user = User(email=email)
                    self.db.add(user)
                    self.db.commit()
                
                # Create new session
                session = DBSession(
                    id=session_id,
                    email=email,
                    stage="initial",
                    analysis_complete=False,
                    ai_state={}
                )
                self.db.add(session)
                self.db.commit()

            # Initialize or get conversation state
            if session_id not in self.conversation_state:
                # Try to load existing state from database
                if session.ai_state:
                    try:
                        self.conversation_state[session_id] = ConversationState.from_dict(session.ai_state)
                    except Exception as e:
                        logger.error(f"Error loading state from database: {str(e)}")
                        # If loading fails, create new state
                        self.conversation_state[session_id] = ConversationState(
                            stage="initial",
                            demographics={},
                            symptoms={},
                            diagnostic_tests={},
                            treatments={},
                            lifestyle={},
                            chat_history=[],
                            title="New MS Consultation",
                            analysis_complete=False,
                            analysis={},
                            recommendations={}
                        )
                else:
                    # Create new state
                    self.conversation_state[session_id] = ConversationState(
                        stage="initial",
                        demographics={},
                        symptoms={},
                        diagnostic_tests={},
                        treatments={},
                        lifestyle={},
                        chat_history=[],
                        title="New MS Consultation",
                        analysis_complete=False,
                        analysis={},
                        recommendations={}
                    )

            state = self.conversation_state[session_id]
            
            # Add message to chat history
            state.chat_history.append({"role": "user", "content": message})
            
            # Process message and get response
            response = self._get_stage_response(state, message)
            state.chat_history.append({"role": "assistant", "content": response})
            
            # Convert state to dict for database storage
            state_dict = state.to_dict()
            
            # Update session in database
            session.stage = state.stage
            session.analysis_complete = state.analysis_complete
            session.ai_state = state_dict
            session.last_updated = datetime.utcnow()
            self.db.commit()
            
            return response
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except StateError as e:
            logger.error(f"State error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise MSHealthAIError(f"Failed to process message: {str(e)}")

    def _get_stage_response(self, state: ConversationState, message: str) -> str:
        """Get the appropriate response based on the current stage."""
        current_stage = state.stage
        
        if current_stage == "initial":
            return self._handle_initial_stage(state, message)
        elif current_stage == "demographics":
            return self._handle_demographics_stage(state, message)
        elif current_stage == "symptoms":
            return self._handle_symptoms_stage(state, message)
        elif current_stage == "diagnostic_tests":
            return self._handle_diagnostic_tests_stage(state, message)
        elif current_stage == "treatments":
            return self._handle_treatments_stage(state, message)
        elif current_stage == "lifestyle":
            return self._handle_lifestyle_stage(state, message)
        else:
            return "I'm not sure how to proceed. Could you please provide more information?"

    def _update_session_title(self, state: Dict):
        """Update session title based on chat context and gathered information."""
        if not state["chat_history"]:
            return
        
        # Get the most recent user messages
        recent_messages = [msg["content"] for msg in state["chat_history"][-6:] if msg["role"] == "user"]
        if not recent_messages:
            return
        
        # Generate title based on symptoms if available
        if state["symptoms"]:
            main_symptoms = list(state["symptoms"].keys())[:2]
            title = f"MS Consultation: {', '.join(main_symptoms)}"
        # Generate title based on recent messages
        else:
            # Take the first meaningful message
            for msg in recent_messages:
                if len(msg.strip()) > 10:
                    title = msg[:50].strip()
                    if len(msg) > 50:
                        title += "..."
                    break
            else:
                title = "New MS Consultation"
        
        state["title"] = title

    def _handle_initial_stage(self, state: ConversationState, message: str) -> str:
        """Handle the initial stage of the conversation."""
        message = message.lower().strip()
        
        # Check if this is a follow-up question about previous analysis
        if state.analysis_complete:
            return self._handle_analysis_stage(state, message)
        
        # Handle greetings and casual conversation
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(greeting in message for greeting in greetings):
            # Check if we have previous conversation
            if state.chat_history:
                return "Hello again! How can I help you today? Would you like to continue our previous discussion or start a new assessment?"
            
            state.stage = "demographics"
            return "Hello! I'm your MS Health Assistant. I'm here to help you understand and manage your condition better. To get started, could you tell me your age and gender?"
        
        # Handle questions about MS
        ms_questions = ["what is ms", "what is multiple sclerosis", "tell me about ms", "explain ms"]
        if any(q in message for q in ms_questions):
            return "Multiple Sclerosis (MS) is a chronic disease affecting the central nervous system. It occurs when the immune system attacks the protective covering of nerve fibers, causing communication problems between the brain and the rest of the body. Would you like to start an assessment to better understand your specific situation?"
        
        # Handle requests for help or guidance
        help_requests = ["help", "guide", "what can you do", "how can you help"]
        if any(req in message for req in help_requests):
            return "I can help you in several ways:\n1. Assess your symptoms and provide personalized insights\n2. Track your condition over time\n3. Provide information about treatments and lifestyle management\n4. Answer your questions about MS\n\nWould you like to start with an assessment? If so, please share your age and gender."
        
        # If it's not a greeting or specific question, move to demographics and try to parse
        state.stage = "demographics"
        return self._handle_demographics_stage(state, message)

    def _handle_demographics_stage(self, state: ConversationState, message: str) -> str:
        """Handle the demographics stage of the conversation."""
        try:
            # Check if we already have demographics
            if state.demographics:
                # If we have both age and gender, move to symptoms
                if "age" in state.demographics and "gender" in state.demographics:
                    state.stage = "symptoms"
                    return f"Thank you for providing your information. Now, could you tell me about any symptoms you're experiencing? Please describe them in detail - such as fatigue, numbness, vision problems, or any other concerns you may have."
                # If we're missing one piece of information, ask for it specifically
                elif "age" not in state.demographics:
                    return "I see you've provided your gender. Could you please tell me your age?"
                elif "gender" not in state.demographics:
                    return "I see you've provided your age. Could you please tell me your gender (male/female/other)?"

            # Try to parse new demographics
            demographics = self._parse_demographics(message)
            state.demographics.update(demographics)
            
            # Check what information we have and what we need
            has_age = "age" in state.demographics
            has_gender = "gender" in state.demographics
            
            if not has_age and not has_gender:
                return "To help you better, I need to know your age and gender. Could you please provide both? For example: 'I am 35 years old and male' or just '35, male'."
            elif not has_age:
                return "Thank you for providing your gender. Could you please tell me your age?"
            elif not has_gender:
                return "Thank you for providing your age. Could you please tell me your gender (male/female/other)?"
            
            # If we have both age and gender, move to symptoms
            state.stage = "symptoms"
            return f"Thank you for providing your information. Now, could you tell me about any symptoms you're experiencing? Please describe them in detail - such as fatigue, numbness, vision problems, or any other concerns you may have."
            
        except Exception as e:
            logger.error(f"Error in demographics stage: {str(e)}")
            return "I'm having trouble understanding. Could you please provide your age and gender? For example: 'I am 35 years old and male'."

    def _handle_symptoms_stage(self, state: ConversationState, message: str) -> str:
        """Handle the symptoms stage of the conversation."""
        try:
            # Check if user is asking about previous symptoms
            if "what symptoms" in message.lower() or "what did i say" in message.lower():
                if state.symptoms:
                    response = "Here are the symptoms you've mentioned so far:\n"
                    for category in ["physical", "cognitive", "emotional"]:
                        if state.symptoms.get(category):
                            response += f"\n{category.title()} symptoms:\n"
                            for symptom in state.symptoms[category]:
                                response += f"- {symptom}\n"
                    response += "\nAre you experiencing any other symptoms?"
                    return response
                else:
                    return "You haven't mentioned any symptoms yet. Could you please describe any symptoms you're experiencing?"

            # Parse new symptoms
            symptoms = self._parse_symptoms(message)
            
            # Update symptoms in state
            for category, symptom_list in symptoms.items():
                if symptom_list:  # Only update if there are symptoms
                    if category not in state.symptoms:
                        state.symptoms[category] = []
                    # Add only new symptoms
                    for symptom in symptom_list:
                        if symptom not in state.symptoms[category]:
                            state.symptoms[category].append(symptom)
            
            # Generate response based on symptoms mentioned
            response = "Thank you for sharing these symptoms. "
            
            # Add specific responses for common symptoms
            if "fatigue" in message.lower() or "tired" in message.lower():
                response += "I understand you're experiencing significant fatigue. This is a common symptom in MS that can be quite debilitating. "
            if "numb" in message.lower() or "tingling" in message.lower():
                response += "The numbness and tingling sensations you're experiencing could be related to nerve damage. "
            if "forgetful" in message.lower() or "memory" in message.lower():
                response += "The cognitive changes you're noticing, including forgetfulness, are also common in MS. "
            if "blurry" in message.lower() or "vision" in message.lower():
                response += "Even mild vision changes can be significant. "
            if "mood" in message.lower() or "depressed" in message.lower() or "blah" in message.lower():
                response += "The mood changes you're experiencing could be related to both the physical symptoms and the impact of MS on your life. "
            
            # Check if we have enough symptom information
            total_symptoms = sum(len(state.symptoms.get(cat, [])) for cat in ["physical", "cognitive", "emotional"])
            
            if total_symptoms == 0:
                return "Could you tell me more about any symptoms you're experiencing? For example: fatigue, numbness, memory problems, depression, vision issues, or any other concerns."
            
            # Ask about specific categories that haven't been mentioned
            missing_categories = []
            for category in ["physical", "cognitive", "emotional"]:
                if not state.symptoms.get(category):
                    missing_categories.append(category)
            
            if missing_categories:
                if "physical" in missing_categories:
                    response += "Are you experiencing any other physical symptoms like muscle weakness or coordination problems? "
                if "cognitive" in missing_categories:
                    response += "Any cognitive issues like memory problems or difficulty concentrating? "
                if "emotional" in missing_categories:
                    response += "Any emotional changes like depression or anxiety?"
                return response
            
            # If we have comprehensive symptom information, move to diagnostic tests
            state.stage = "diagnostic_tests"
            response += "\n\nBased on the symptoms you've described, it would be helpful to know if you've had any diagnostic tests done. Have you had any MRI scans, blood tests, or other medical tests?"
            return response
            
        except Exception as e:
            logger.error(f"Error in symptoms stage: {str(e)}")
            return "I'm having trouble understanding your symptoms. Could you please describe them in more detail? For example: 'I experience fatigue and numbness in my hands'."

    def _handle_diagnostic_tests_stage(self, state: ConversationState, message: str) -> str:
        """Handle the diagnostic tests stage of the conversation."""
        try:
            # Check if user is asking about previous tests
            if "what tests" in message.lower() or "what did i say" in message.lower():
                if state.diagnostic_tests:
                    response = "Here are the tests you've mentioned so far:\n"
                    for test_name, test_info in state.diagnostic_tests.items():
                        if test_name != "none":
                            response += f"- {test_info['name']}: {', '.join(test_info.get('findings', ['Performed']))}\n"
                    response += "\nHave you had any other tests done?"
                    return response
                else:
                    return "You haven't mentioned any tests yet. Have you had any diagnostic tests done, such as MRI scans or blood tests?"

            # Parse new tests
            tests = self._parse_diagnostic_tests(message)
            
            # Update tests in state
            for test_name, test_info in tests.items():
                if test_name not in state.diagnostic_tests:
                    state.diagnostic_tests[test_name] = test_info
            
            # Check if we have test information
            has_tests = len(state.diagnostic_tests) > 0 and "none" not in state.diagnostic_tests
            
            if not has_tests:
                if "no" in message.lower() or "none" in message.lower():
                    # User hasn't had tests
                    state.diagnostic_tests["none"] = {"name": "No tests performed", "findings": []}
                    state.stage = "treatments"
                    return "I understand you haven't had diagnostic tests yet. That's okay. Are you currently taking any medications or receiving any treatments for your symptoms?"
                else:
                    return "Have you had any diagnostic tests done? Such as MRI scans, blood tests, or other medical examinations? If not, please say 'no' or 'none'."
            
            # Generate response based on tests mentioned
            response = "Thank you for sharing your test information. "
            
            # Add specific responses for different tests
            if "mri" in state.diagnostic_tests:
                response += "I see you've had an MRI scan. "
                if "lesion" in str(state.diagnostic_tests["mri"].get("findings", [])).lower():
                    response += "The presence of lesions is an important finding in MS diagnosis. "
                elif "normal" in str(state.diagnostic_tests["mri"].get("findings", [])).lower():
                    response += "A normal MRI is good news, though it doesn't completely rule out MS. "
            
            if "blood_tests" in state.diagnostic_tests:
                response += "You've also had blood tests done. "
                if "normal" in str(state.diagnostic_tests["blood_tests"].get("findings", [])).lower():
                    response += "Normal blood test results help rule out other conditions. "
            
            if "spinal_tap" in state.diagnostic_tests:
                response += "The spinal tap results can provide important information about MS. "
                if "oligoclonal" in str(state.diagnostic_tests["spinal_tap"].get("findings", [])).lower():
                    response += "The presence of oligoclonal bands supports an MS diagnosis. "
            
            # Move to treatments stage
            state.stage = "treatments"
            response += "\n\nNow, could you tell me about any medications or treatments you're currently taking for your symptoms?"
            return response
            
        except Exception as e:
            logger.error(f"Error in diagnostic tests stage: {str(e)}")
            return "I'm having trouble understanding the test information. Could you please provide more details about any tests you've had, or say 'none' if you haven't had any tests?"

    def _handle_treatments_stage(self, state: ConversationState, message: str) -> str:
        """Handle the treatments stage of the conversation."""
        try:
            # Check if user is asking about previous treatments
            if "what treatments" in message.lower() or "what medications" in message.lower():
                if state.treatments:
                    response = "Here are the treatments you've mentioned so far:\n"
                    if state.treatments.get("current"):
                        response += "\nCurrent treatments:\n"
                        for treatment in state.treatments["current"]:
                            response += f"- {treatment}\n"
                    if state.treatments.get("past"):
                        response += "\nPast treatments:\n"
                        for treatment in state.treatments["past"]:
                            response += f"- {treatment}\n"
                    response += "\nAre you taking any other medications or treatments?"
                    return response
                else:
                    return "You haven't mentioned any treatments yet. Are you currently taking any medications or receiving any treatments for your symptoms?"

            # Parse new treatments
            treatments = self._parse_treatments(message)
            
            # Update treatments in state
            if treatments.get("current"):
                if "current" not in state.treatments:
                    state.treatments["current"] = []
                for treatment in treatments["current"]:
                    if treatment not in state.treatments["current"]:
                        state.treatments["current"].append(treatment)
            
            if treatments.get("past"):
                if "past" not in state.treatments:
                    state.treatments["past"] = []
                for treatment in treatments["past"]:
                    if treatment not in state.treatments["past"]:
                        state.treatments["past"].append(treatment)
            
            # Check if we have treatment information
            has_treatments = (state.treatments.get("current") and len(state.treatments["current"]) > 0) or \
                           "no" in message.lower() or "none" in message.lower()
            
            if not has_treatments:
                return "Could you tell me what medications you're currently taking for MS or your symptoms? If you're not taking any, please say 'none'."
            
            # If no treatments, record that
            if "no" in message.lower() or "none" in message.lower():
                state.treatments["current"] = ["None"]
            
            # Move to lifestyle stage
            state.stage = "lifestyle"
            
            # Generate response based on treatments mentioned
            response = "Thank you for the treatment information. "
            if state.treatments.get("current") and "None" not in state.treatments["current"]:
                response += "I see you're currently taking some medications. "
            elif state.treatments.get("current") and "None" in state.treatments["current"]:
                response += "I understand you're not currently taking any medications. "
            
            response += "Now, could you tell me about your lifestyle? This includes your diet, exercise routine, and how you manage stress."
            return response
            
        except Exception as e:
            logger.error(f"Error in treatments stage: {str(e)}")
            return "I'm having trouble understanding your treatment information. Could you please tell me about any medications you're taking, or say 'none' if you're not taking any?"

    def _handle_lifestyle_stage(self, state: ConversationState, message: str) -> str:
        """Handle the lifestyle stage of the conversation."""
        try:
            # Check if user is asking about previous lifestyle information
            if "what lifestyle" in message.lower() or "what did i say" in message.lower():
                if state.lifestyle:
                    response = "Here's what you've told me about your lifestyle:\n"
                    for category, details in state.lifestyle.items():
                        if category != "general":
                            response += f"\n{category.replace('_', ' ').title()}:\n"
                            for detail in details:
                                response += f"- {detail}\n"
                    response += "\nIs there anything else you'd like to add about your lifestyle?"
                    return response
                else:
                    return "You haven't shared any lifestyle information yet. Could you tell me about your diet, exercise routine, and how you manage stress?"

            # Parse new lifestyle information
            lifestyle = self._parse_lifestyle(message)
            
            # Update lifestyle in state
            for category, details in lifestyle.items():
                if category not in state.lifestyle:
                    state.lifestyle[category] = []
                for detail in details:
                    if detail not in state.lifestyle[category]:
                        state.lifestyle[category].append(detail)
            
            # Check if we have lifestyle information
            has_lifestyle = len(state.lifestyle) > 0 and "general" not in state.lifestyle
            
            if not has_lifestyle:
                return "Could you tell me about your lifestyle? For example, what kind of diet do you follow, what exercise do you do, and how do you manage stress?"
            
            # Move to analysis stage and generate analysis
            state.stage = "analysis"
            state.analysis = self._generate_analysis(state)
            state.recommendations = self._generate_recommendations(state)
            state.analysis_complete = True
            
            # Generate response based on lifestyle information
            response = "Thank you for sharing your lifestyle information. "
            if "diet" in state.lifestyle:
                response += "I see you've shared some information about your diet. "
            if "exercise" in state.lifestyle:
                response += "You've also mentioned your exercise routine. "
            if "stress_management" in state.lifestyle:
                response += "And you've told me about your stress management strategies. "
            
            response += f"\n\nBased on all the information you've provided, here's my analysis:\n\n{state.analysis}\n\nRecommendations:\n{state.recommendations}"
            return response
            
        except Exception as e:
            logger.error(f"Error in lifestyle stage: {str(e)}")
            return "I'm having trouble understanding your lifestyle information. Could you please provide more details about your diet, exercise, and stress management?"

    def _handle_analysis_stage(self, state: ConversationState, message: str) -> str:
        """Handle the analysis stage of the conversation."""
        if not state.analysis:
            state.analysis = self._generate_analysis(state)
            state.recommendations = self._generate_recommendations(state)
            state.analysis_complete = True
        
        # This is follow-up conversation after analysis
        return "Is there anything specific about the analysis or recommendations you'd like me to explain further?"

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a session.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Returns:
            Optional[Dict[str, Any]]: Session state if found, None otherwise
            
        Raises:
            ValidationError: If session_id is invalid
        """
        if not session_id or not isinstance(session_id, str):
            raise ValidationError("Invalid session ID")
            
        # Try to get state from memory first
        if session_id in self.conversation_state:
            return self.conversation_state[session_id].to_dict()
            
        # If not in memory, try to get from database
        session = self.db.query(DBSession).filter(DBSession.id == session_id).first()
        if session and session.ai_state:
            return session.ai_state
            
        return None

    def clear_session(self, session_id: str) -> None:
        """
        Clear a session's state.
        
        Args:
            session_id: Unique identifier for the conversation session
            
        Raises:
            ValidationError: If session_id is invalid
        """
        if not session_id or not isinstance(session_id, str):
            raise ValidationError("Invalid session ID")
            
        # Clear from memory
        if session_id in self.conversation_state:
            del self.conversation_state[session_id]
            
        # Clear from database
        session = self.db.query(DBSession).filter(DBSession.id == session_id).first()
        if session:
            session.ai_state = {}
            session.stage = "initial"
            session.analysis_complete = False
            session.last_updated = datetime.utcnow()
            self.db.commit()

    def _validate_state(self, state: ConversationState) -> None:
        """
        Validate the conversation state.
        
        Args:
            state: Conversation state to validate
            
        Raises:
            StateError: If state is invalid
        """
        if not isinstance(state, ConversationState):
            raise StateError("Invalid state type")
            
        if not state.stage or state.stage not in [
            "initial", "demographics", "symptoms", 
            "diagnostic_tests", "treatments", "lifestyle", "analysis"
        ]:
            raise StateError("Invalid stage")
            
        if not isinstance(state.demographics, dict):
            raise StateError("Invalid demographics")
            
        if not isinstance(state.symptoms, dict):
            raise StateError("Invalid symptoms")
            
        if not isinstance(state.diagnostic_tests, dict):
            raise StateError("Invalid diagnostic tests")
            
        if not isinstance(state.treatments, dict):
            raise StateError("Invalid treatments")
            
        if not isinstance(state.lifestyle, dict):
            raise StateError("Invalid lifestyle")
            
        if not isinstance(state.chat_history, list):
            raise StateError("Invalid chat history")
            
        if not isinstance(state.title, str):
            raise StateError("Invalid title")

    def _update_state(self, session_id: str, state: ConversationState) -> None:
        """
        Update the conversation state with validation.
        
        Args:
            session_id: Unique identifier for the conversation session
            state: New conversation state
            
        Raises:
            ValidationError: If session_id is invalid
            StateError: If state is invalid
        """
        if not session_id or not isinstance(session_id, str):
            raise ValidationError("Invalid session ID")
            
        self._validate_state(state)
        self.conversation_state[session_id] = state
        
        # Update database with dictionary representation
        session = self.db.query(DBSession).filter(DBSession.id == session_id).first()
        if session:
            session.stage = state.stage
            session.analysis_complete = state.analysis_complete
            session.ai_state = state.to_dict()
            session.last_updated = datetime.utcnow()
            self.db.commit()

    def _parse_demographics(self, message: str) -> Dict:
        try:
            demographics = {}
            message = message.lower().strip()
            
            # Extract age - more flexible patterns
            import re
            
            # Try different age patterns
            age_patterns = [
                r'\b(\d{1,2})\b',  # Any 1-2 digit number
                r'i am (\d+)', r'i\'m (\d+)', r'age (\d+)',
                r'(\d+) years', r'(\d+) year', r'(\d+) y/o',
                r'(\d+) years old', r'(\d+) year old'
            ]
            
            for pattern in age_patterns:
                match = re.search(pattern, message)
                if match:
                    age = int(match.group(1))
                    if 1 <= age <= 120:  # Reasonable age range
                        demographics["age"] = age
                        break
            
            # Extract gender - more flexible matching
            if any(word in message for word in ["male", "man", "boy", "m"]):
                demographics["gender"] = "male"
            elif any(word in message for word in ["female", "woman", "girl", "f"]):
                demographics["gender"] = "female"
            elif any(word in message for word in ["non-binary", "nonbinary", "other", "nb"]):
                demographics["gender"] = "non-binary"
            
            return demographics
        except Exception as e:
            logger.error(f"Error parsing demographics: {str(e)}")
            return {}

    def _parse_symptoms(self, message: str) -> Dict:
        try:
            symptoms = {"physical": [], "cognitive": [], "emotional": []}
            message = message.lower()
            
            # Check physical symptoms
            physical_keywords = {
                "fatigue": ["fatigue", "tired", "exhausted", "energy", "wiped out"],
                "numbness": ["numbness", "numb", "tingling", "pins and needles"],
                "muscle weakness": ["weakness", "weak", "muscle", "strength"],
                "vision problems": ["vision", "sight", "eye", "blurry", "blurred"],
                "balance problems": ["balance", "unsteady", "dizzy", "vertigo"],
                "pain": ["pain", "ache", "hurt", "sore"],
                "walking difficulties": ["walking", "walk", "mobility", "gait"],
                "coordination problems": ["coordination", "clumsy", "uncoordinated"],
                "tremors": ["tremor", "shaking", "trembling"],
                "spasticity": ["spasticity", "stiff", "rigid", "tight"],
                "bladder problems": ["bladder", "urination", "incontinence"],
                "bowel problems": ["bowel", "constipation", "diarrhea"],
                "sexual dysfunction": ["sexual", "libido", "erection", "orgasm"]
            }
            
            for symptom, keywords in physical_keywords.items():
                if any(keyword in message for keyword in keywords):
                    symptoms["physical"].append(symptom)
            
            # Check cognitive symptoms
            cognitive_keywords = {
                "memory problems": ["memory", "forget", "remember", "recall"],
                "difficulty concentrating": ["concentration", "focus", "attention", "distracted"],
                "brain fog": ["fog", "cloudy", "confused", "fuzzy"],
                "information processing": ["processing", "slow", "thinking", "thought"],
                "executive function": ["planning", "organization", "decision", "judgment"],
                "visual-spatial problems": ["spatial", "depth", "distance", "judge"]
            }
            
            for symptom, keywords in cognitive_keywords.items():
                if any(keyword in message for keyword in keywords):
                    symptoms["cognitive"].append(symptom)
            
            # Check emotional symptoms
            emotional_keywords = {
                "depression": ["depression", "depressed", "sad", "down", "low"],
                "anxiety": ["anxiety", "anxious", "worry", "nervous", "stress"],
                "mood swings": ["mood", "irritable", "emotional", "moody"],
                "emotional lability": ["lability", "emotional", "mood changes"],
                "stress": ["stress", "stressed", "overwhelmed"],
                "irritability": ["irritable", "irritated", "angry", "frustrated"]
            }
            
            for symptom, keywords in emotional_keywords.items():
                if any(keyword in message for keyword in keywords):
                    symptoms["emotional"].append(symptom)
            
            return symptoms
        except Exception as e:
            logger.error(f"Error parsing symptoms: {str(e)}")
            return {"physical": [], "cognitive": [], "emotional": []}

    def _parse_diagnostic_tests(self, message: str) -> Dict:
        try:
            tests = {}
            message = message.lower()
            
            # Check for MRI
            if "mri" in message:
                tests["mri"] = {
                    "name": "Magnetic Resonance Imaging (MRI)",
                    "findings": []
                }
                if "lesion" in message:
                    tests["mri"]["findings"].append("Lesions detected")
                elif "normal" in message:
                    tests["mri"]["findings"].append("Normal")
                else:
                    tests["mri"]["findings"].append("Results mentioned")
            
            # Check for blood tests
            if any(term in message for term in ["blood test", "blood work", "blood"]):
                tests["blood_tests"] = {
                    "name": "Blood Tests",
                    "findings": []
                }
                if "normal" in message:
                    tests["blood_tests"]["findings"].append("Normal")
                else:
                    tests["blood_tests"]["findings"].append("Results mentioned")
            
            return tests
        except Exception as e:
            logger.error(f"Error parsing diagnostic tests: {str(e)}")
            return {}

    def _parse_treatments(self, message: str) -> Dict:
        try:
            treatments = {"current": [], "past": []}
            message = message.lower()
            
            # Check for common MS medications
            ms_medications = [
                "interferon", "copaxone", "glatiramer", "tecfidera", "dimethyl fumarate",
                "gilenya", "fingolimod", "tysabri", "natalizumab", "ocrevus", "ocrelizumab"
            ]
            
            for med in ms_medications:
                if med in message:
                    treatments["current"].append(med.title())
            
            # If no specific medications found but treatment mentioned
            if not treatments["current"] and any(word in message for word in ["medication", "drug", "treatment", "taking"]):
                treatments["current"].append("Unspecified medication")
            
            return treatments
        except Exception as e:
            logger.error(f"Error parsing treatments: {str(e)}")
            return {"current": [], "past": []}

    def _parse_lifestyle(self, message: str) -> Dict:
        try:
            lifestyle = {}
            message = message.lower()
            
            # Diet keywords
            diet_keywords = ["diet", "eat", "food", "nutrition"]
            if any(keyword in message for keyword in diet_keywords):
                lifestyle["diet"] = ["Diet mentioned"]
            
            # Exercise keywords
            exercise_keywords = ["exercise", "workout", "gym", "walk", "run", "sport"]
            if any(keyword in message for keyword in exercise_keywords):
                lifestyle["exercise"] = ["Exercise mentioned"]
            
            # Stress management keywords
            stress_keywords = ["stress", "relax", "meditation", "yoga"]
            if any(keyword in message for keyword in stress_keywords):
                lifestyle["stress_management"] = ["Stress management mentioned"]
            
            # If nothing specific mentioned, assume basic lifestyle
            if not lifestyle:
                lifestyle["general"] = ["Basic lifestyle"]
            
            return lifestyle
        except Exception as e:
            logger.error(f"Error parsing lifestyle: {str(e)}")
            return {"general": ["Basic lifestyle"]}

    def _generate_analysis(self, state: Dict) -> str:
        try:
            analysis = "Based on the information provided, here's my analysis:\n\n"
            
            # Demographics
            if state["demographics"]:
                analysis += "Patient Profile:\n"
                if "age" in state["demographics"]:
                    analysis += f"- Age: {state['demographics']['age']}\n"
                if "gender" in state["demographics"]:
                    analysis += f"- Gender: {state['demographics']['gender'].title()}\n"
                analysis += "\n"
            
            # Symptoms
            if state["symptoms"]:
                analysis += "Symptom Analysis:\n"
                for category in ["physical", "cognitive", "emotional"]:
                    if state["symptoms"].get(category):
                        analysis += f"- {category.title()} symptoms: {', '.join(state['symptoms'][category])}\n"
                analysis += "\n"
            
            # Diagnostic tests
            if state["diagnostic_tests"]:
                analysis += "Diagnostic Information:\n"
                for test_name, test_info in state["diagnostic_tests"].items():
                    if test_name != "none":
                        analysis += f"- {test_info['name']}: {', '.join(test_info.get('findings', ['Performed']))}\n"
                    else:
                        analysis += "- No diagnostic tests performed yet\n"
                analysis += "\n"
            
            # Current treatments
            if state["treatments"]:
                analysis += "Treatment Status:\n"
                if state["treatments"].get("current"):
                    if "None" in state["treatments"]["current"]:
                        analysis += "- No current treatments\n"
                    else:
                        analysis += f"- Current treatments: {', '.join(state['treatments']['current'])}\n"
                analysis += "\n"
            
            return analysis
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return "Analysis could not be generated due to an error."

    def _generate_recommendations(self, state: Dict) -> str:
        try:
            recommendations = "\n"
            
            # General recommendations
            recommendations += "1. Schedule regular follow-ups with a neurologist specializing in MS\n"
            recommendations += "2. Keep a detailed symptom diary to track changes over time\n"
            recommendations += "3. Consider joining an MS support group for emotional support\n"
            
            # Specific recommendations based on symptoms
            if state["symptoms"]:
                recommendations += "\nSymptom-specific recommendations:\n"
                
                # Physical symptoms
                if state["symptoms"].get("physical"):
                    recommendations += "- For physical symptoms: Consider physical therapy and regular low-impact exercise\n"
                
                # Cognitive symptoms
                if state["symptoms"].get("cognitive"):
                    recommendations += "- For cognitive symptoms: Practice mental exercises and consider cognitive rehabilitation\n"
                
                # Emotional symptoms
                if state["symptoms"].get("emotional"):
                    recommendations += "- For emotional symptoms: Consider counseling or therapy support\n"
            
            # Diagnostic recommendations
            if not state["diagnostic_tests"] or "none" in state["diagnostic_tests"]:
                recommendations += "\nDiagnostic recommendations:\n"
                recommendations += "- Consider getting an MRI scan to evaluate for MS lesions\n"
                recommendations += "- Blood tests to rule out other conditions\n"
                recommendations += "- Consultation with a neurologist for comprehensive evaluation\n"
            
            # Treatment recommendations
            if not state["treatments"].get("current") or "None" in state["treatments"].get("current", []):
                recommendations += "\nTreatment considerations:\n"
                recommendations += "- Discuss disease-modifying therapies with your neurologist\n"
                recommendations += "- Consider symptom management strategies\n"
            
            # Lifestyle recommendations
            recommendations += "\nLifestyle recommendations:\n"
            recommendations += "- Maintain a healthy, anti-inflammatory diet\n"
            recommendations += "- Regular exercise as tolerated\n"
            recommendations += "- Stress management techniques (meditation, yoga)\n"
            recommendations += "- Adequate sleep and rest\n"
            recommendations += "- Vitamin D supplementation (consult with doctor)\n"
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return "Recommendations could not be generated due to an error."

    def _parse_mycotoxin_tests(self, message: str) -> Dict:
        """Parse mycotoxin test results from user message."""
        try:
            tests = {}
            message = message.lower()
            
            # Check for each mycotoxin test
            for test_key, test_info in self.knowledge_base["mycotoxin_tests"].items():
                test_name = test_info["name"].lower()
                if test_name in message:
                    # Extract the value
                    import re
                    value_match = re.search(r'(\d+\.?\d*)', message)
                    if value_match:
                        value = float(value_match.group(1))
                        # Determine result category
                        if value < float(test_info["reference_ranges"]["not_present"].replace("<", "")):
                            result = "not_present"
                        elif value < float(test_info["reference_ranges"]["equivocal"].split(" to <")[1]):
                            result = "equivocal"
                        else:
                            result = "present"
                        
                        tests[test_key] = {
                            "name": test_info["name"],
                            "value": value,
                            "result": result,
                            "reference_ranges": test_info["reference_ranges"]
                        }
            
            return tests
        except Exception as e:
            logger.error(f"Error parsing mycotoxin tests: {str(e)}")
            return {}

    def _analyze_mycotoxin_results(self, test_results: Dict) -> str:
        """Analyze mycotoxin test results and provide comprehensive insights."""
        try:
            analysis = "Mycotoxin Test Analysis:\n\n"
            
            if not test_results:
                return "No mycotoxin test results provided."
            
            # Analyze each test result
            for test_key, result in test_results.items():
                test_info = self.knowledge_base["mycotoxin_tests"][test_key]
                analysis += f"{test_info['name']}:\n"
                analysis += f"- Value: {result['value']}\n"
                analysis += f"- Result: {result['result'].replace('_', ' ').title()}\n"
                analysis += f"- Mechanism: {test_info['activity']}\n"
                
                # Add interpretation based on result
                if result['result'] == 'present':
                    analysis += "- Interpretation: Elevated levels detected. "
                    if test_info['symptoms']:
                        analysis += f"Common associated symptoms include: {', '.join(test_info['symptoms'])}. "
                    if test_info['disease_states']:
                        analysis += f"May be associated with: {', '.join(test_info['disease_states'])}.\n"
                    if test_info['treatment_considerations']:
                        analysis += f"Treatment considerations: {', '.join(test_info['treatment_considerations'])}.\n"
                elif result['result'] == 'equivocal':
                    analysis += "- Interpretation: Borderline levels detected. Consider retesting in 2-3 weeks.\n"
                else:
                    analysis += "- Interpretation: Levels within normal range.\n"
                
                analysis += "\n"
            
            # Add interaction analysis
            elevated_tests = [test for test in test_results.values() if test['result'] == 'present']
            if len(elevated_tests) > 1:
                analysis += "Toxin Interaction Analysis:\n"
                for interaction in self.knowledge_base["mycotoxin_interactions"]["synergistic"]:
                    toxins = interaction.split(" + ")
                    if all(any(t.lower() in test_info["name"].lower() for test_info in test_results.values()) for t in toxins):
                        analysis += f"- Potential synergistic interaction: {interaction}\n"
            
            # Add overall assessment
            if elevated_tests:
                analysis += "\nOverall Assessment:\n"
                analysis += "- Multiple elevated mycotoxin levels detected. "
                analysis += "This may indicate exposure to mold or other environmental toxins. "
                analysis += "Consider consulting with a healthcare provider specializing in environmental medicine.\n\n"
                
                # Add treatment recommendations
                analysis += "Treatment Recommendations:\n"
                for category, protocols in self.knowledge_base["treatment_protocols"].items():
                    analysis += f"\n{category.title()}:\n"
                    for protocol in protocols:
                        analysis += f"- {protocol}\n"
            elif any(test['result'] == 'equivocal' for test in test_results.values()):
                analysis += "\nOverall Assessment:\n"
                analysis += "- Some borderline results detected. Consider retesting in 2-3 weeks.\n"
            else:
                analysis += "\nOverall Assessment:\n"
                analysis += "- All mycotoxin levels are within normal range.\n"
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing mycotoxin results: {str(e)}")
            return "Error analyzing mycotoxin test results."

    def _handle_mycotoxin_stage(self, state: Dict, message: str) -> str:
        """Handle the mycotoxin testing stage of the conversation."""
        try:
            # Check if user is asking about previous test results
            if "what tests" in message.lower() or "what results" in message.lower():
                if state.get("mycotoxin_tests"):
                    response = "Here are your mycotoxin test results:\n\n"
                    for test_name, test_info in state["mycotoxin_tests"].items():
                        response += f"{test_info['name']}: {test_info['value']} ({test_info['result'].replace('_', ' ').title()})\n"
                    response += "\nWould you like me to analyze these results in detail?"
                    return response
                else:
                    return "You haven't provided any mycotoxin test results yet. Please share your test results, including the values for each test."

            # Parse new test results
            test_results = self._parse_mycotoxin_tests(message)
            
            if not test_results:
                return "I couldn't find any mycotoxin test results in your message. Please provide the test results with their values. For example: 'Ochratoxin A: 2.1' or 'Aflatoxin Group: 0.9'."

            # Update test results in state
            if "mycotoxin_tests" not in state:
                state["mycotoxin_tests"] = {}
            state["mycotoxin_tests"].update(test_results)
            
            # Generate analysis
            analysis = self._analyze_mycotoxin_results(state["mycotoxin_tests"])
            
            # Move to next stage
            state["stage"] = "analysis"
            state["analysis"] = analysis
            state["analysis_complete"] = True
            
            return f"Thank you for providing your mycotoxin test results. Here's my analysis:\n\n{analysis}\n\nWould you like me to explain any specific aspects of these results in more detail?"
            
        except Exception as e:
            logger.error(f"Error in mycotoxin stage: {str(e)}")
            return "I'm having trouble understanding your test results. Could you please provide them in a clear format? For example: 'Ochratoxin A: 2.1' or 'Aflatoxin Group: 0.9'."