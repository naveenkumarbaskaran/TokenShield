# TokenShield Architecture

## System Overview

```mermaid
flowchart LR
    subgraph App["Your Application"]
        A[LLM Call] --> B[Shield.call]
    end

    subgraph TokenShield
        B --> C{Estimator}
        C -->|tokens| D{Budget Gate}
        D -->|ALLOW| E[LLM API]
        D -->|REJECT| F[BudgetExceeded]
        E -->|response| G[Tracker]
        G --> H{Alert Check}
        H -->|threshold hit| I[Alert Hook]
        H --> J[Return Result]
    end

    style F fill:#ff6b6b,color:#fff
    style D fill:#ffd93d,color:#333
    style G fill:#6bcb77,color:#fff
```

## Request Lifecycle

```mermaid
sequenceDiagram
    participant App
    participant Shield
    participant Estimator
    participant BudgetGate
    participant LLM
    participant Tracker
    participant AlertHook

    App->>Shield: call(messages, tools)
    Shield->>Estimator: estimate_message_tokens()
    Estimator-->>Shield: ~1,200 tokens
    
    Shield->>BudgetGate: check(estimated_cost=$0.003)
    
    alt Budget OK
        BudgetGate-->>Shield: ALLOW
        Shield->>LLM: completion(messages, tools)
        LLM-->>Shield: response + usage
        Shield->>Tracker: record(cost=$0.003)
        Tracker->>AlertHook: check thresholds
        
        alt Threshold Hit
            AlertHook-->>App: ⚠️ 80% of daily budget
        end
        
        Shield-->>App: result
    else Budget Exceeded
        BudgetGate-->>Shield: REJECT
        Shield-->>App: BudgetExceeded exception
    end
```

## Cost Tracking Data Flow

```mermaid
flowchart TD
    subgraph Input
        M[Messages] --> T1[Token Count]
        TL[Tools] --> T2[Token Count]
    end

    subgraph Estimation
        T1 --> SUM[Total Input Tokens]
        T2 --> SUM
        SUM --> COST["Cost = tokens × price / 1M"]
    end

    subgraph Tracking
        COST --> REC[RequestRecord]
        REC --> TODAY[cost_today]
        REC --> HOUR[cost_last_hour]
        REC --> MODEL[cost_by_model]
    end

    subgraph Export
        TODAY --> CSV[CSV Export]
        TODAY --> JSON[JSON Export]
        TODAY --> REPORT[Dashboard Report]
    end

    style COST fill:#ffd93d,color:#333
    style REC fill:#6bcb77,color:#fff
```

## Budget Gate Decision Tree

```mermaid
flowchart TD
    START[Incoming Request] --> EST[Estimate Cost]
    EST --> CHK1{Per-request\nlimit set?}
    
    CHK1 -->|Yes| CMP1{est > limit?}
    CHK1 -->|No| CHK2
    
    CMP1 -->|Yes| REJECT[❌ BudgetExceeded\nper-request]
    CMP1 -->|No| CHK2{Per-hour\nlimit set?}
    
    CHK2 -->|Yes| CMP2{hour_cost +\nest > limit?}
    CHK2 -->|No| CHK3
    
    CMP2 -->|Yes| REJECT2[❌ BudgetExceeded\nper-hour]
    CMP2 -->|No| CHK3{Per-day\nlimit set?}
    
    CHK3 -->|Yes| CMP3{day_cost +\nest > limit?}
    CHK3 -->|No| ALLOW
    
    CMP3 -->|Yes| REJECT3[❌ BudgetExceeded\nper-day]
    CMP3 -->|No| ALLOW[✅ Execute LLM Call]

    style REJECT fill:#ff6b6b,color:#fff
    style REJECT2 fill:#ff6b6b,color:#fff
    style REJECT3 fill:#ff6b6b,color:#fff
    style ALLOW fill:#6bcb77,color:#fff
```

## Cost Comparison: With vs Without TokenShield

```mermaid
xychart-beta
    title "Monthly LLM Cost ($) — With vs Without TokenShield"
    x-axis ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"]
    y-axis "Cost (USD)" 0 --> 4000
    bar [50, 200, 3400, 3800, 4200, 5000]
    line [50, 180, 220, 250, 280, 310]
```

| Metric | Without Shield | With Shield | Savings |
|--------|---------------|-------------|---------|
| Month 6 cost | $5,000 | $310 | **94%** |
| Runaway incidents | 3 | 0 | **100%** |
| Avg cost/request | $0.12 | $0.03 | **75%** |
| Budget overruns | 4 | 0 | **100%** |
