#!/usr/bin/env python3
"""
Semantic Kernel AI Agent Core
Initializes and manages an AI agent powered by Microsoft Semantic Kernel
"""
#TODO: fix that sometimes ticker is not actually used for the interaction
#TODO: investigate that sometimes datavalidation function is called multiple
#  times and analyze if it's actually repeating unnecessary validations
#TODO: summarize history to prevent excesive token input

import os
from openai import AsyncOpenAI
from semantic_kernel import Kernel, kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from data_processing import DataProcessingPlugin
from market_analysis import MarketAnalysisPlugin

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from your .env file into os.e

class AIAgentCore:
    """Core AI Agent powered by Semantic Kernel"""
    def __init__(self):
        """Initialize the Semantic Kernel and AI Agent"""
        self.kernel = self._initialize_kernel()
        self.settings = OpenAIPromptExecutionSettings(service_id="custom_chat_service")
        self.settings.function_choice_behavior = FunctionChoiceBehavior.Auto(auto_invoke=True)
        self.system_message = ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content = 
                "You are a helpful market research assistant. Use your available tools " \
                "to fetch/validate data, generate market summarizations and generate basic trend analysis based on the user's intent." \
                "Ensure tickers you use to make fetch_data request are well formed english-alphabetic strings"
                "Always validate data right after fetching it and remember that you already validated to prevent repeated validations on same data" \
                "When doing market summarizations always do it through semantic function called market_summarization"
                "When doing trends identification do it through semantic function called basic_trends_identification"
        )
        #TODO: add chat history prunning
        self.chat_history = ChatHistory()
        self.chat_history.messages.insert(0, self.system_message)
        self._setup_plugins()



    def _setup_plugins(self):
        """Register plugins with the kernel"""
        self.data_processing_plugin = DataProcessingPlugin()
        self.kernel.add_plugin(self.data_processing_plugin, plugin_name="DataProcessingPlugin")

        self.market_analysis_plugin = MarketAnalysisPlugin()
        self.kernel.add_plugin(self.market_analysis_plugin, plugin_name="MarketAnalysisPlugin")

    def _initialize_kernel(self) -> Kernel:
        """
        Initialize the Semantic Kernel with Azure OpenAI
        
        Returns:
            Kernel: Configured Semantic Kernel instance
        """
        kernel = Kernel()

        my_custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        openai_chat_service = OpenAIChatCompletion(
            service_id="custom_chat_service",
            ai_model_id="gpt-4o",
            async_client=my_custom_client,
        )

        kernel.add_service(openai_chat_service)

        return kernel

    async def invoke_agent(self, user_input: str) -> str:
        """
        Invoke the AI agent with user input
        
        Args:
            user_input (str): User query or command
            
        Returns:
            str: Agent response
        """
        chat_completion = self.kernel.get_service("custom_chat_service")

        print("going to get result")

        self.chat_history.add_user_message(user_input)

        result = await chat_completion.get_chat_message_content(
            chat_history=self.chat_history,
            settings=self.settings,
            kernel = self.kernel,
        )   

        self.chat_history.add_message(result)

        result_content = str(result.content).strip()

        return result_content

    async def run_agent_loop(self):
        """Run interactive agent loop"""
        print("🤖 Semantic Kernel AI Agent Started")
        print("Type 'exit' to quit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Agent: Goodbye! 👋")
                    break
                
                if not user_input:
                    continue

                response = await self.invoke_agent(user_input)
                print(f"Agent: {response}\n")

            except KeyboardInterrupt:
                print("\nAgent: Session interrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

async def main():
    """Main entry point"""
    agent = AIAgentCore()
    await agent.run_agent_loop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())