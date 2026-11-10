# Graph Feasibility and Fraud-Ring Analysis

## Objective

This document evaluates whether graph-based analysis can improve fraud detection by identifying suspicious relationships between accounts, transactions, and simulated enrichment entities.

The goal is not only to classify an individual transaction as fraudulent, but also to determine whether an account is connected to other suspicious accounts or transactions.

## Why graph analysis matters

Traditional transaction-level models evaluate each transaction independently. They can identify unusual amounts, frequencies, or patterns, but they may miss coordinated fraud where multiple accounts behave normally in isolation.

Graph analysis adds relationship context:

```text
Account A → Transaction → Account B
Account B → Transaction → Account C
Account C → Transaction → Account D
```

If several connected accounts participate in suspicious transactions, the connected component may represent a fraud ring even when no single transaction looks obviously fraudulent.

## PaySim graph suitability

PaySim provides sender and receiver account identifiers, transaction type, amount, event time, and fraud labels. These fields support a transaction graph.

### Available graph entities

| Entity | Source field | Role |
|---|---|---|
| Sender account | `sender_id` | Graph node |
| Receiver account | `receiver_id` | Graph node |
| Transaction | `transaction_id` | Edge or event node |
| Fraud label | `is_fraud` | Node/edge attribute |
| Transaction type | `type` | Edge attribute |
| Amount | `amount` | Edge attribute |
| Event time | `event_time` | Temporal filtering attribute |

## Graph design

### Nodes

- **Account node:** Represents a sender or receiver account.
- **Transaction node:** Represents an individual transaction event.

### Edges

- `SENT`
  - From sender account to transaction.
- `RECEIVED`
  - From transaction to receiver account.
- `TRANSFERRED_TO`
  - From sender account to receiver account, derived from a transaction.

### Edge attributes

- Transaction amount.
- Transaction type.
- Event time.
- Fraud label.
- Model fraud probability, where available.

## Temporal safety

Graph features must respect the same leakage rule as tabular features:

```text
For a transaction at time T, use only transactions strictly before T.
```

The transaction being scored must never contribute to its own graph features.

### Safe graph features

- Number of prior transactions by sender.
- Number of prior transactions by receiver.
- Number of distinct counterparties in the recent window.
- Recent transaction amount statistics.
- Number of prior fraud-connected accounts.
- Number of prior suspicious counterparties.
- Recent graph degree.
- Recent shared-counterparty count.

### Unsafe graph features

These must not be used for real-time scoring:

- Current transaction fraud label.
- Current transaction amount used as historical evidence.
- Current receiver relationship used as a prior relationship.
- Future transactions.
- Post-transaction balance fields that reveal the transaction outcome.

## Fraud-ring detection approach

### 1. Build account-transaction graph

Create a graph where accounts are nodes and transactions connect them.

```text
Sender account
→ Transaction
→ Receiver account
```

### 2. Filter by time window

For each transaction, construct graph features using only earlier transactions.

Example:

```text
Current transaction time: T
Allowed historical window: T - 24 hours to T
Excluded: transaction at T and all transactions after T
```

### 3. Identify suspicious components

Flag connected components where:

- Multiple accounts participate in fraud-labeled transactions.
- Accounts share counterparties within a short period.
- Transaction amounts or frequencies show coordinated behavior.
- A model-scored transaction connects to prior fraud evidence.

### 4. Generate explanations

For an alert, provide evidence such as:

- Number of connected suspicious accounts.
- Number of prior fraud transactions in the component.
- Shared counterparties.
- Recent transaction velocity.
- Amount deviation from account history.
- Relevant graph paths.

Example explanation:

```text
High risk because the sender is connected to 3 accounts involved in prior
fraud-labeled transactions within the last 24 hours. The current transaction
also shows an unusually high amount compared with the sender's historical
activity.
```

## Feasibility assessment

| Capability | Feasibility | Reason |
|---|---|---|
| Account relationship graph | High | PaySim provides sender and receiver IDs |
| Transaction graph | High | Each transaction can be represented as an edge or event node |
| Temporal graph features | High | Event time supports historical-window filtering |
| Fraud-ring identification | Medium to high | Fraud labels and account connectivity support suspicious-component analysis |
| Real-time graph scoring | Medium | Requires careful feature precomputation and low-latency lookup |
| Explainability | High | Graph paths and connected fraud evidence can be shown to analysts |
| Production scaling | Medium | Requires graph storage, incremental updates, and efficient queries |

## Implementation result

The project implemented graph-based feasibility analysis and explainable relationship checks. It demonstrated that account connectivity can provide additional fraud context beyond individual transaction features.

The implementation supports:

- Account-to-account relationship construction.
- Historical graph feature logic.
- Suspicious-connection analysis.
- Fraud-ring feasibility assessment.
- Human-readable graph explanations.

## Limitations

- PaySim does not include real device, IP, merchant, or location identifiers.
- Device, IP, and location fields used for enrichment are deterministic simulated fields derived from account identifiers.
- PaySim fraud labels are synthetic and may not fully represent real-world fraud-ring behavior.
- Real production graph scoring would require a graph database or optimized in-memory graph service.

## Conclusion

Graph analysis is feasible and valuable for this project. It adds relationship-based context that a transaction-only model cannot capture, supports explainable fraud-ring evidence, and improves the platform’s fraud-investigation narrative.

The recommended production approach is to combine:

```text
Tabular transaction features
+ Historical behavioral features
+ Temporal graph features
+ XGBoost model score
+ Graph explanation
```

This produces a more defensible fraud decision than a standalone classifier.