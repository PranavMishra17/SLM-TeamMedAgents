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
    """Hugging Face chat instance using transformers library."""
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
            import torch
            self.AutoProcessor = AutoProcessor
            self.AutoModelForImageTextToText = AutoModelForImageTextToText
            self.pipeline = pipeline
            self.torch = torch
        except ImportError:
            raise ImportError("transformers library not available. Install with: pip install transformers torch")
        
        # Check HF token
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
        
        self.hf_model_name = model_config.get("hf_model_name", model_config["model"])
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        
        # Initialize model and processor
        self._initialize_model()
        
        logging.info(f"Initialized {self.display_name} via Hugging Face on {self.device}")
    
    def _initialize_model(self):
        """Initialize the Hugging Face model and processor."""
        try:
            if self.supports_vision:
                # For vision models
                self.processor = self.AutoProcessor.from_pretrained(
                    self.hf_model_name,
                    token=os.environ.get('HUGGINGFACE_TOKEN')
                )
                self.model = self.AutoModelForImageTextToText.from_pretrained(
                    self.hf_model_name,
                    token=os.environ.get('HUGGINGFACE_TOKEN'),
                    torch_dtype=self.torch.float16 if self.device == "cuda" else self.torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                # For text-only models, use pipeline
                self.pipe = self.pipeline(
                    "text-generation",
                    model=self.hf_model_name,
                    token=os.environ.get('HUGGINGFACE_TOKEN'),
                    torch_dtype=self.torch.float16 if self.device == "cuda" else self.torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
        except Exception as e:
            logging.error(f"Failed to initialize Hugging Face model: {e}")
            raise
    
    def simple_chat(self, message: str, image_path: str = None) -> str:
        """Simple single-turn chat with optional image support."""
        try:
            if self.supports_vision:
                return self._vision_chat(message, image_path)
            else:
                return self._text_chat(message)
        except Exception as e:
            logging.error(f"Error in Hugging Face simple_chat: {e}")
            raise
    
    def _vision_chat(self, message: str, image_path: str = None) -> str:
        """Handle vision-enabled chat."""
        try:
            content = [{"type": "text", "text": message}]
            
            if image_path:
                image = self._load_image(image_path)
                if image:
                    # Convert PIL image to URL format expected by processor
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    image_url = f"data:image/png;base64,{image_data}"
                    content.insert(0, {"type": "image", "url": image_url})
            
            messages = [{"role": "user", "content": content}]
            
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            max_new_tokens = self.model_config.get("max_tokens", 512)
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=self.model_config.get("temperature", 0.3),
                do_sample=True if self.model_config.get("temperature", 0.3) > 0 else False
            )
            
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
        except Exception as e:
            logging.error(f"Error in vision chat: {e}")
            raise
    
    def _text_chat(self, message: str) -> str:
        """Handle text-only chat."""
        try:
            outputs = self.pipe(
                message,
                max_new_tokens=self.model_config.get("max_tokens", 512),
                temperature=self.model_config.get("temperature", 0.3),
                do_sample=True if self.model_config.get("temperature", 0.3) > 0 else False,
                return_full_text=False
            )
            
            return outputs[0]['generated_text'].strip()
        except Exception as e:
            logging.error(f"Error in text chat: {e}")
            raise
    
    def streaming_chat(self, message: str, image_path: str = None):
        """Streaming chat response generator."""
        # For now, return the full response as a single chunk
        # Hugging Face streaming requires more complex setup
        response = self.simple_chat(message, image_path)
        yield response
    
    def conversation_chat(self, messages: List[Dict[str, str]], image_paths: List[str] = None) -> str:
        """Multi-turn conversation chat."""
        # For now, use the last message
        # Full conversation support requires conversation templates
        last_message = messages[-1]["content"] if messages else ""
        last_image = image_paths[-1] if image_paths and image_paths else None
        
        return self.simple_chat(last_message, last_image)

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
