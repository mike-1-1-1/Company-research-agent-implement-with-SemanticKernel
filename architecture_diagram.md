# Plugin Architecture Diagram

```mermaid
flowchart TD
    A[User input / request] --> B[Data fetch and validation plugin]
    B --> C[Normalized market dataset]
    C --> D[Market analysis plugin]
    D --> E[Summary + trend detection]
    E --> F[Output / report]

    subgraph Plugins[Plugin boundary]
        B
        D
    end

    subgraph Extensions[Future extensions]
        G[Reporting plugin]
        H[Charting / visualization plugin]
        I[Additional analytics plugins]
    end

    D --> G
    D --> H
    D --> I
    G --> F
    H --> F
    I --> F
```

## Notes
- The Data Processing plugin is responsible for ingesting, validating, and shaping raw market data.
- The Market Analysis plugin focuses on insight generation such as summaries and trend detection.
- Additional plugins can be attached later for reporting, visualization, or specialized analytics without changing the core flow.
