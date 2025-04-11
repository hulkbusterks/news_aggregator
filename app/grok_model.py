import os
from groq import Groq
from google.colab import userdata

class GroqAgent:
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key is None:
                api_key = userdata.get('GROQ_API_KEY')
                if api_key is None:
                    raise ValueError("GROQ_API_KEY not set.")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate_text(self, prompt, max_tokens=1024, temperature=0.7, top_p=1.0, top_k=50, stream=False):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return None

    def chat(self, messages, max_tokens=1024, temperature=0.7, top_p=1.0, top_k=50, stream=False):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error during chat: {e}")
            return None
  
    def summarize(self, text, mode="default"):
        try:
            prompt = f"Summarize the following text:\n\n{text}"
            if mode == "5-year-old":
                prompt = f"Summarize the following text for a 5-year-old:\n\n{text}"
            elif mode == "detailed":
                prompt = f"Provide a detailed summary of the following text:\n\n{text}"
            elif mode == "layman":
                prompt = f"Summarize the following text in simple terms:\n\n{text}"
            summary = self.generate_text(prompt)
            return summary
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return None

    def explain(self, text, mode="default"):
        try:
            prompt = f"Explain the following text:\n\n{text}"
            if mode == "5-year-old":
                prompt = f"Explain the following text to a 5-year-old:\n\n{text}"
            elif mode == "detailed":
                prompt = f"Provide a detailed explanation of the following text:\n\n{text}"
            elif mode == "layman":
                prompt = f"Explain the following text in simple terms:\n\n{text}"
            explanation = self.generate_text(prompt)
            return explanation
        except Exception as e:
            print(f"Error explaining text: {e}")
            return None

    def translate(self, text, target_language="en"):
        try:
            prompt = f"Translate the following text to {target_language}:\n\n{text}"
            translation = self.generate_text(prompt)
            return translation
        except Exception as e:
            print(f"Error translating text: {e}")
            return None
          
if __name__ == "__main__":
  groq_agent = GroqAgent() 
  # Example usage and summary/explanation testing
  text_to_process = "This is a test string for demonstration.  It includes several sentences to showcase the summarization and explanation capabilities.  Let's add a few more sentences to make it a bit more substantial. And even one more for good measure."
  
  
  # Test different modes of summarize function
  print("Summarization tests:")
  print("Default:", groq_agent.summarize(text_to_process))
  print("5-year-old:", groq_agent.summarize(text_to_process, mode="5-year-old"))
  print("Detailed:", groq_agent.summarize(text_to_process, mode="detailed"))
  print("Layman:", groq_agent.summarize(text_to_process, mode="layman"))
  
  # Test different modes of explain function
  print("\nExplanation tests:")
  print("Default:", groq_agent.explain(text_to_process))
  print("5-year-old:", groq_agent.explain(text_to_process, mode="5-year-old"))
  print("Detailed:", groq_agent.explain(text_to_process, mode="detailed"))
  print("Layman:", groq_agent.explain(text_to_process, mode="layman"))
  
  response = groq_agent.generate_text("Write a short story about a cat.")
  if response:
      print("\nShort story about a cat:\n", response)
  
  messages = [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I am doing well, thank you."},
      {"role": "user", "content": "What is the capital of France? and what did i said"},
  ]
  chat_response = groq_agent.chat(messages)
  if chat_response:
      print("\nChat response:\n", chat_response)
    
  #Test translate with diff languages
  print(groq_agent.translate("Hello, world!", target_language="hi")) # Hindi
  print(groq_agent.translate("How are you?", target_language="fr")) # French
  print(groq_agent.translate("What is your name?", target_language="es")) # Spanish
  print(groq_agent.translate("Good morning", target_language="de")) # German
  print(groq_agent.translate("Good night", target_language="ru")) # Russian
  print(groq_agent.translate("Where are you from?", target_language="it")) # Italian
  print(groq_agent.translate("This is a test", target_language="ja")) # Japanese
  print(groq_agent.translate("Thank you", target_language="ko")) # Korean
  print(groq_agent.translate("You're welcome", target_language="pt")) # Portuguese
  print(groq_agent.translate("Goodbye", target_language="zh-CN")) # Chinese
