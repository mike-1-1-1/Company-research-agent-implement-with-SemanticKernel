import unittest
import sys
from pathlib import Path
import asyncio

# 1. Get directories
PROJECT_ROOT = Path(__file__).resolve().parent
PLUGINS_DIR = PROJECT_ROOT / "plugins"

# 2. Append BOTH to sys.path before importing anything else
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PLUGINS_DIR))

# 3. Now the imports will resolve successfully
from plugins.data_processing import DataProcessingPlugin
from plugins.market_analysis import MarketAnalysisPlugin
from plugins.main import AIAgentCore


class PluginIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_fetch_data_handles_structured_payload(self) -> None:
        stocksTicker = "AAPL"
        from_ = "2026-05-22"
        to = "2026-07-22"

        data_plugin = DataProcessingPlugin()
        fetched_data = data_plugin.fetch_data(stocksTicker, from_, to)
        self.assertTrue(fetched_data)

    async def test_market_data_extraction_then_market_summarization_and_finally_trend_analysis(self) -> None:
        agent = AIAgentCore()

        data_extraction_result = await agent.invoke_agent("Get the financial data of Nvidia from Jan 2026 to March 2026")

        self.assertTrue(data_extraction_result)
        self.assertTrue(len(data_extraction_result) > 10)
        print(data_extraction_result)

        await asyncio.sleep(3)

        market_summarization = await agent.invoke_agent("Perform a market summarization")
        self.assertTrue(market_summarization)
        self.assertTrue(len(market_summarization) > 10)
        print(market_summarization)
        self.assertIn("nvidia", market_summarization.lower())
        self.assertIn("summar", market_summarization.lower())

        await asyncio.sleep(3)

        trend_analysis = await agent.invoke_agent("Provide a trend analysis")

        self.assertTrue(trend_analysis)
        self.assertTrue(len(trend_analysis) > 10)
        print(trend_analysis)
        self.assertIn("nvidia", trend_analysis.lower())
        self.assertIn("trend", trend_analysis.lower())
        


if __name__ == "__main__":
    unittest.main()
