from __future__ import annotations

from typing import Any, Dict, List

from semantic_kernel import Kernel, kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from polygon import RESTClient as MassiveRESTClient
import os

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from your .env file into os.e

"""
Data Processing Plugin - Native Functions
Handles data fetching and validation.

Future Enhancement Opportunities:
- Statistical Analysis: Use pandas for time-series analysis, correlation studies,
  and aggregation of market metrics. pandas DataFrames would enable efficient
  grouping, pivoting, and statistical operations on large datasets.

- Chart Generation: Integrate matplotlib for creating line charts (price trends),
  bar charts (volume analysis), and candlestick charts. Use seaborn for enhanced
  statistical visualizations like heatmaps (correlation matrices), distribution
  plots, and regression analysis with confidence intervals.
"""

class DataProcessingPlugin:
    """Native plugin responsible for getting and validating market data."""

    def __init__(self):
        self.massiveClient = MassiveRESTClient(api_key=os.getenv("MASSIVE_API_KEY"))

    def get_stock_ticker(self, kernel: Kernel):
        config_parameters_prompt = """
            You are a financial assistant. Given the company name or description, 
            return only the official stock ticker symbol (e.g., AAPL for Apple). 
            Do not include any extra text or punctuation.
            Company: {{$company_name}} Ticker:"""
        print('executing get_stock_ticker function')  
        return kernel.add_function(
            function_name="GetStockTicker",
            plugin_name="DataProcessing",
            prompt=config_parameters_prompt,
            description="Get stock ticker from natural language company name."
        )

    @kernel_function(
        name="FetchMarketData",
        description="Fetches raw market data from API"
    )
    def fetch_data(self, stocks_ticker, from_, to) -> Dict[str, Any]:
        print("executing fetch_data function")
        #TODO : maybe limit to a monthly basis
        try:
            aggs = self.massiveClient.get_aggs(
                ticker=stocks_ticker,
                multiplier=1,
                timespan="month",
                from_=from_,
                to=to,
                adjusted=True,
            )
        except Exception as e:
            error = f'"Massive" API error: {e}'
            print(error)
            aggs = error

        # if not isinstance(payload, dict):
        #     raise TypeError("Payload must be a dictionary")

        # market = str(payload.get("market", "")).strip()
        # if not market:
        #     raise ValueError("A market name is required")

        # records = payload.get("records", [])
        # if not isinstance(records, list) or not records:
        #     raise ValueError("At least one record is required")

        # normalized_records: List[Dict[str, Any]] = []
        # for record in records:
        #     if not isinstance(record, dict):
        #         raise TypeError("Each record must be a dictionary")

        #     normalized = {
        #         "date": str(record.get("date", "")).strip(),
        #         "revenue": float(record.get("revenue", 0)),
        #         "volume": float(record.get("volume", 0)),
        #     }
        #     normalized_records.append(normalized)

        # return {"market": market, "records": normalized_records}
        if not aggs:
            raise ValueError("At least one result is needed")

        data = {'metrics': aggs}
        
        #print('aggs', aggs)
        print('data', data)
        return data

    @kernel_function(
        name="ValidateMarketData",
        description="Fetches raw market data from API"
    )
    def validate_data(self, data: Dict[str, Any]):
        print("executing validate_data function")
        """
        Validates the structure and content of data.

        This function ensures that the input data is in the expected format
        (dictionary) and contains all required fields. It performs type checking
        and field presence validation before the data is used in analysis.

        Args:
            data: The data to validate (expected to be a dictionary)

        Returns:
            dict: The validated data if all checks pass

        Raises:
            ValueError: If data is not a dictionary or required fields are missing
        """

        # Check that data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data).__name__}")

        # Validate that metrics is a dictionary
        if not data['metrics']:
            raise ValueError("Field 'metrics' must be not empty")

        # Define required fields for market data
        required_fields = ["open", "high", "low"]

        # Check presence of each required field
        missing_fields = []
        for agg in data['metrics']:
            for field in required_fields:
                if field not in agg:
                    missing_fields.append(field)

        # Raise error if any required fields are missing
        if missing_fields:
            raise ValueError(f"Missing required field(s): {', '.join(missing_fields)}")

        # If all validations pass, return data
        return data