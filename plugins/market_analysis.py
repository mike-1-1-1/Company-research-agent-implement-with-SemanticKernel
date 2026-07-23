from __future__ import annotations

from typing import Any, Dict, List

from semantic_kernel import Kernel
# from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter


class MarketAnalysisPlugin:
    """Semantic-style plugin that summarizes market conditions and trends."""        

    def market_summarization(self, kernel: Kernel):
        print("executing market_summarization function")
        """Create a Semantic Kernel prompt-based market summarization function.

        Args:
            kernel (Kernel): The Semantic Kernel instance used to register the function.

        Returns:
            Any: The registered Semantic Kernel function object returned by the kernel.
        """
        config_parameters_prompt = """
        You are a market analysis assistant that returns a dictionary with key findings based on
        analyzing upward/downward/stagnant trends in sales, summarizing dominant sector and alerting on major changes. 
        Take the input dataset dictionary 
        [DATA_START]
        {{$market_dataset_dict}}
        [DATA_END]

        Respond ONLY in the following JSON format:
        {
        "summary": a summary of key findings
        "trend_direction": "upward" or "downward" or "stagnant",
        "dominant_sector": e.g. "Technology sector" or "industry_average",
        "major changes": a possibly empty list of major changes
        }
        """

        return kernel.add_function(
            function_name="MarketSummarization",
            plugin_name="MarketAnalysis",
            prompt=config_parameters_prompt,
            description="Does a market summarization analysis based on the input dataset"
        )

    def basic_trends_identification(self, kernel: Kernel):
        print("executing basic_trends_identification function")
        """Create a Semantic Kernel prompt-based trend identification function.

        Args:
            kernel (Kernel): The Semantic Kernel instance used to register the function.

        Returns:
            Any: The registered Semantic Kernel function object returned by the kernel.
        """

        config_parameters_prompt = """
        You are an analysis assistant that returns a dictionary with key trend findings such as (but no limited to)
        growth/decline, seasonal patterns, anomalies, cultural shifts, major events (natural disasters, conflicts, etc.).
        Take the input dataset dictionary
        [DATA_START]
        {{$market_dataset_dict}}
        [DATA_END] 
        Respond ONLY in the following JSON format:
        {
        "summary": a summary of key trends found
        "trends": an array of of the trends found with arguments of why they were found
        }
        """

        return kernel.add_function(
            function_name="BasicTrendsIdentification",
            plugin_name="MarketAnalysis",
            prompt=config_parameters_prompt,
            description="Does a trend analysis based on the input dataset"
        )


       
