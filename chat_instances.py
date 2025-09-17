"""
Modular chat instances for different model providers.
Supports Google AI Studio, Hugging Face, and future extensibility.
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import base64
from PIL import Image
import io

# Environment setup
from dotenv import load_dotenv
load_dotenv()

class BaseChatInstance(ABC):
    """Abstract base class for chat instances."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.display_name = model_config["display_name"]
        self.supports_vision = model_config.get("supports_vision", False)
        
    @abstractmethod
    def simple_chat(self, message: str, image_path: str = None) -> str:
        """Simple single-turn chat with optional image support."""
        pass
    
    @abstractmethod
    def streaming_chat(self, message: str, image_path: str = None):
        """Streaming chat response generator."""
        pass
    
    @abstractmethod
    def conversation_chat(self, messages: List[Dict[str, str]], image_paths: List[str] = None) -> str:
        """Multi-turn conversation chat."""
        pass
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Helper to load and validate image."""
        if not image_path or not os.path.exists(image_path):
            logging.warning(f"Image file not found: {image_path}")
            return None
        
        try:
            return Image.open(image_path)
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {e}")
            return None

class GoogleAIStudioChatInstance(BaseChatInstance):
    """Google AI Studio chat instance using google-genai library."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
        except ImportError:
            raise ImportError("google-genai library not available. Install with: pip install google-genai")
        
        # Initialize client
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model_config["model"]
        
        logging.info(f"Initialized {self.display_name} via Google AI Studio")
    
    def simple_chat(self, message: str, image_path: str = None) -> str:
        """Simple single-turn chat with optional image support."""
        try:
            parts = [self.types.Part.from_text(text=message)]
            
            # Add image if provided and model supports vision
            if image_path and self.supports_vision:
                image = self._load_image(image_path)
                if image:
                    # Convert to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    parts.append(self.types.Part.from_uri(
                        file_uri=f"data:image/png;base64,{image_data}",
                        mime_type="image/png"
                    ))
            
            contents = [self.types.Content(role="user", parts=parts)]
            
            config = self.types.GenerateContentConfig(
                temperature=self.model_config.get("temperature", 0.3),
                max_output_tokens=self.model_config.get("max_tokens", 8192)
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            return response.text
        except Exception as e:
            logging.error(f"Error in Google AI Studio simple_chat: {e}")
            raise
    
    def streaming_chat(self, message: str, image_path: str = None):
        """Streaming chat response generator."""
        try:
            parts = [self.types.Part.from_text(text=message)]
            
            if image_path and self.supports_vision:
                image = self._load_image(image_path)
                if image:
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    parts.append(self.types.Part.from_uri(
                        file_uri=f"data:image/png;base64,{image_data}",
                        mime_type="image/png"
                    ))
            
            contents = [self.types.Content(role="user", parts=parts)]
            
            config = self.types.GenerateContentConfig(
                temperature=self.model_config.get("temperature", 0.3),
                max_output_tokens=self.model_config.get("max_tokens", 8192)
            )
            
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            ):
                yield chunk.text
        except Exception as e:
            logging.error(f"Error in Google AI Studio streaming_chat: {e}")
            raise
    
    def conversation_chat(self, messages: List[Dict[str, str]], image_paths: List[str] = None) -> str:
        """Multi-turn conversation chat."""
        try:
            contents = []
            for i, msg in enumerate(messages):
                role = "user" if msg["role"] == "user" else "model"
                parts = [self.types.Part.from_text(text=msg["content"])]
                
                # Add image if provided for this message
                if (image_paths and i < len(image_paths) and image_paths[i] and 
                    self.supports_vision and msg["role"] == "user"):
                    image = self._load_image(image_paths[i])
                    if image:
                        buffer = io.BytesIO()
                        image.save(buffer, format='PNG')
                        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        parts.append(self.types.Part.from_uri(
                            file_uri=f"data:image/png;base64,{image_data}",
                            mime_type="image/png"
                        ))
                
                contents.append(self.types.Content(role=role, parts=parts))
            
            config = self.types.GenerateContentConfig(
                temperature=self.model_config.get("temperature", 0.3),
                max_output_tokens=self.model_config.get("max_tokens", 8192)
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            
            return response.text
        except Exception as e:
            logging.error(f"Error in Google AI Studio conversation_chat: {e}")
            raise

class HuggingFaceChatInstance(BaseChatInstance):
    """Hugging Face chat instance using transformers library with pipeline API."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        
        try:
            import torch
            from transformers import pipeline, BitsAndBytesConfig
            self.torch = torch
            self.pipeline = pipeline
            self.BitsAndBytesConfig = BitsAndBytesConfig
        except ImportError:
            raise ImportError("Required packages not available. Install with: pip install transformers torch accelerate bitsandbytes")
        
        # Check HF token
        hf_token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN or HF_TOKEN environment variable not set")
        
        self.hf_model_name = model_config.get("hf_model_name", model_config["model"])
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        
        # Determine if this is a text-only or multimodal model
        self.is_text_only = "text" in self.hf_model_name.lower()
        
        # Initialize model using pipeline API
        self._initialize_pipeline()
        
        logging.info(f"Initialized {self.display_name} via Hugging Face Pipeline on {self.device}")
    
    def _initialize_pipeline(self):
        """Initialize the Hugging Face pipeline with optimized settings."""
        try:
            # Configure quantization for large models
            use_quantization = self._should_use_quantization()
            
            # Remove token from model_kwargs to avoid duplicate parameter error
            model_kwargs = {
                "torch_dtype": self.torch.bfloat16,
                "device_map": "auto"
            }
            
            if use_quantization:
                model_kwargs["quantization_config"] = self.BitsAndBytesConfig(load_in_4bit=True)
            
            # For multimodal models, always initialize text-generation pipeline to avoid torch version issues
            # Only use image-text-to-text when actually processing images
            task = "text-generation"
            logging.info(f"Loading {self.hf_model_name} as text-generation pipeline")
            
            # Initialize pipeline with token passed separately
            self.pipe = self.pipeline(
                task,
                model=self.hf_model_name,
                model_kwargs=model_kwargs,
                token=os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
            )
            
            # Configure generation settings
            self.pipe.model.generation_config.do_sample = False
            
        except Exception as e:
            logging.error(f"Failed to initialize Hugging Face pipeline: {e}")
            raise
    
    def _should_use_quantization(self) -> bool:
        """Determine if quantization should be used based on model size and available memory."""
        # Use quantization for large models (27B+ parameters) or when CUDA memory is limited
        if "27b" in self.hf_model_name.lower():
            return True
        
        # Check available CUDA memory
        if self.torch.cuda.is_available():
            try:
                memory_gb = self.torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if memory_gb < 16:  # Less than 16GB VRAM
                    return True
            except:
                pass
        
        return False
    
    def simple_chat(self, message: str, image_path: str = None) -> str:
        """Simple single-turn chat with optional image support."""
        try:
            # Use text-only mode if no image provided, even for multimodal models
            if image_path and os.path.exists(image_path) and not self.is_text_only:
                return self._multimodal_chat(message, image_path)
            else:
                return self._text_chat(message)
        except Exception as e:
            logging.error(f"Error in Hugging Face simple_chat: {e}")
            raise
    
    def _text_chat(self, message: str) -> str:
        """Handle text-only chat using pipeline."""
        try:
            # For text generation pipeline, pass the message directly as string
            max_new_tokens = self.model_config.get("max_tokens", 512)
            
            outputs = self.pipe(
                message,  # Pass message directly, not as 'text='
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False
            )
            
            return outputs[0]['generated_text'].strip()
        except Exception as e:
            logging.error(f"Error in text chat: {e}")
            raise
    
    def _multimodal_chat(self, message: str, image_path: str = None) -> str:
        """Handle multimodal chat with image support."""
        try:
            # Prepare system message
            role_instruction = "You are an expert medical AI assistant."
            system_instruction = role_instruction
            
            # Prepare user content
            user_content = [{"type": "text", "text": message}]
            
            # Add image if provided
            if image_path and os.path.exists(image_path):
                try:
                    from PIL import Image
                    image = Image.open(image_path)
                    user_content.append({"type": "image", "image": image})
                except Exception as e:
                    logging.warning(f"Failed to load image {image_path}: {e}")
            
            # Format messages
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_instruction}]
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            max_new_tokens = self.model_config.get("max_tokens", 300)
            
            # Generate response
            output = self.pipe(text=messages, max_new_tokens=max_new_tokens)
            response = output[0]["generated_text"][-1]["content"]
            
            # Handle thinking mode for 27B models (if applicable)
            if "27b" in self.hf_model_name.lower() and "<unused95>" in response:
                thought, actual_response = response.split("<unused95>")
                response = actual_response
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error in multimodal chat: {e}")
            raise
    
    def streaming_chat(self, message: str, image_path: str = None):
        """Streaming chat response generator."""
        # Pipeline API doesn't easily support streaming, return full response
        response = self.simple_chat(message, image_path)
        yield response
    
    def conversation_chat(self, messages: List[Dict[str, str]], image_paths: List[str] = None) -> str:
        """Multi-turn conversation chat."""
        try:
            # Convert to pipeline format
            formatted_messages = []
            
            for i, msg in enumerate(messages):
                content = [{"type": "text", "text": msg["content"]}]
                
                # Add image if provided for this message
                if (image_paths and i < len(image_paths) and image_paths[i] and 
                    msg["role"] == "user" and not self.is_text_only):
                    try:
                        from PIL import Image
                        image = Image.open(image_paths[i])
                        content.append({"type": "image", "image": image})
                    except Exception as e:
                        logging.warning(f"Failed to load image {image_paths[i]}: {e}")
                
                formatted_messages.append({
                    "role": msg["role"],
                    "content": content
                })
            
            max_new_tokens = self.model_config.get("max_tokens", 512)
            
            output = self.pipe(text=formatted_messages, max_new_tokens=max_new_tokens)
            response = output[0]["generated_text"][-1]["content"]
            
            return response.strip()
            
        except Exception as e:
            logging.error(f"Error in conversation chat: {e}")
            raise

class ChatInstanceFactory:
    """Factory for creating chat instances."""
    
    INSTANCE_TYPES = {
        "google_ai_studio": GoogleAIStudioChatInstance,
        "huggingface": HuggingFaceChatInstance,
    }
    
    @classmethod
    def create_chat_instance(cls, model_config: Dict[str, Any], 
                           instance_type: str = "google_ai_studio") -> BaseChatInstance:
        """Create a chat instance of the specified type."""
        if instance_type not in cls.INSTANCE_TYPES:
            raise ValueError(f"Unknown instance type: {instance_type}. Available: {list(cls.INSTANCE_TYPES.keys())}")
        
        instance_class = cls.INSTANCE_TYPES[instance_type]
        return instance_class(model_config)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available instance types."""
        return list(cls.INSTANCE_TYPES.keys())

# Convenience functions
def create_google_ai_chat(model_config: Dict[str, Any]) -> GoogleAIStudioChatInstance:
    """Create Google AI Studio chat instance."""
    return GoogleAIStudioChatInstance(model_config)

def create_huggingface_chat(model_config: Dict[str, Any]) -> HuggingFaceChatInstance:
    """Create Hugging Face chat instance."""
    return HuggingFaceChatInstance(model_config)

def create_chat_instance(model_config: Dict[str, Any], 
                        instance_type: str = "google_ai_studio") -> BaseChatInstance:
    """Create chat instance using factory."""
    return ChatInstanceFactory.create_chat_instance(model_config, instance_type)