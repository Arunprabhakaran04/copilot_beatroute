# Multi-Agent SQL Query System - Complete Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Core Functionalities](#core-functionalities)
4. [Agent Ecosystem](#agent-ecosystem)
5. [Data Flow](#data-flow)
6. [Key Components](#key-components)
7. [Technology Stack](#technology-stack)
8. [Setup & Configuration](#setup--configuration)

---

## System Overview

### 🎯 Purpose
A production-grade **AI-powered multi-agent orchestration system** that converts natural language queries into executable SQL queries against a CubeJS/PostgreSQL database. The system intelligently decomposes complex queries into multi-step workflows, executes them across specialized agents, and delivers results through real-time WebSocket communication.

### 🌟 Key Capabilities
- **Natural Language to SQL**: Convert English questions into optimized PostgreSQL queries
- **Multi-Step Query Decomposition**: Break complex queries into sequential tasks
- **Conversation Memory**: Redis-backed conversation history with context enrichment
- **Data Visualization**: Automatic chart generation (Plotly) from query results
- **Email Integration**: Send query results and reports via email
- **Meeting Scheduling**: Schedule meetings with parsed date/time extraction
- **Entity Verification**: Validate entity names (SKUs, customers, brands) before query execution
- **Error Recovery**: Automatic SQL error detection and fixing (5 iteration attempts)
- **Real-time Communication**: WebSocket-based streaming responses

---

## Core Architecture

### 🏗️ System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Frontend)                                │
│                    WebSocket Connection (FastAPI)                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   WebSocket Manager     │
                    │  (Session Management)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Authentication Layer   │
                    │  (Token Validation)      │
                    └────────────┬─────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │          CENTRAL ORCHESTRATOR                   │
        │         (LangGraph State Machine)               │
        │                                                  │
        │  Flow: Enrich → Classify → Route → Execute      │
        └────────┬─────────────────────────────┬──────────┘
                 │                             │
    ┌────────────▼────────┐       ┌───────────▼────────────┐
    │   ENRICH AGENT      │       │  REDIS MEMORY MANAGER  │
    │  (Query Enhancement)│       │  (Conversation Cache)  │
    │                     │       │                        │
    │ • Context Analysis  │       │ • Session State        │
    │ • Follow-up Detect  │       │ • Query Results Cache  │
    │ • Entity Resolution │       │ • User Context         │
    └────────┬────────────┘       └───────────┬────────────┘
             │                                │
             └────────────┬───────────────────┘
                          │
         ┌────────────────▼─────────────────┐
         │   AGENT-AWARE DECOMPOSER         │
         │   (Multi-step Detection)         │
         │                                   │
         │ • Single-step vs Multi-step      │
         │ • Task Breakdown                 │
         │ • Agent Sequence Planning        │
         └────────────────┬─────────────────┘
                          │
                 ┌────────▼────────┐
                 │  CLASSIFICATION  │
                 │   (Agent Router) │
                 └────────┬────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────▼─────┐   ┌─────▼──────┐   ┌────▼────────┐
    │ DB QUERY │   │VISUALIZATION│   │   EMAIL     │
    │  AGENT   │   │   AGENT     │   │   AGENT     │
    └────┬─────┘   └─────┬──────┘   └────┬────────┘
         │               │                │
         │          ┌────▼────┐      ┌────▼────┐
         │          │ SUMMARY │      │ MEETING │
         │          │  AGENT  │      │  AGENT  │
         │          └─────────┘      └─────────┘
         │
    ┌────▼──────────────────────────────────────────┐
    │      DB QUERY AGENT (Orchestrator)            │
    │                                                │
    │  ┌──────────────┐  ┌──────────────┐          │
    │  │ SQL QUERY    │  │  IMPROVED    │          │
    │  │ DECOMPOSER   │  │ SQL GENERATOR│          │
    │  └──────┬───────┘  └──────┬───────┘          │
    │         │                 │                   │
    │  ┌──────▼─────────────────▼───────┐          │
    │  │   SQL RETRIEVER AGENT          │          │
    │  │  (Embedding-based Search)      │          │
    │  │                                 │          │
    │  │ • Retrieve similar SQL queries │          │
    │  │ • Top-k similarity search      │          │
    │  │ • Context injection            │          │
    │  └────────────┬────────────────────┘          │
    │               │                               │
    │  ┌────────────▼───────────┐                  │
    │  │  SCHEMA MANAGER        │                  │
    │  │  (Focused Schema)      │                  │
    │  │                        │                  │
    │  │ • Embedding similarity │                  │
    │  │ • Keyword boosting     │                  │
    │  │ • Table extraction     │                  │
    │  └────────────┬───────────┘                  │
    │               │                               │
    │  ┌────────────▼───────────┐                  │
    │  │   SQL GENERATOR        │                  │
    │  │   (LLM-based)          │                  │
    │  └────────────┬───────────┘                  │
    │               │                               │
    │  ┌────────────▼───────────┐                  │
    │  │ SQL EXCEPTION AGENT    │                  │
    │  │ (Error Analysis & Fix) │                  │
    │  │                        │                  │
    │  │ • Pattern matching     │                  │
    │  │ • 5 fix iterations     │                  │
    │  │ • Root cause analysis  │                  │
    │  └────────────┬───────────┘                  │
    └───────────────┼────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   DATABASE LAYER    │
         │  (PostgreSQL)       │
         │                     │
         │ • Connection Pool   │
         │ • Query Execution   │
         │ • DataFrame Results │
         └─────────────────────┘
```

### 🔄 Query Flow Diagram

```
USER QUERY: "Show me top 5 SKUs in September and visualize it"
     │
     ▼
┌────────────────────────────────────────────────┐
│ 1. ENRICH AGENT                                │
│    • Check conversation history                │
│    • Resolve entities (September → 2025-09)    │
│    • Detect multi-step intent                  │
│    Output: "Get top 5 SKUs by sales in         │
│             September 2025 and create chart"   │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ 2. AGENT-AWARE DECOMPOSER                      │
│    • Detect: Multi-step (data + viz)           │
│    • Confidence: 0.95                          │
│    • Plan:                                     │
│      Step 1: db_query (get data)              │
│      Step 2: visualization (create chart)     │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ 3. EXECUTE STEP 1: DB_QUERY AGENT             │
│                                                │
│    3a. SQL RETRIEVER                           │
│        • Search embeddings.pkl                 │
│        • Find similar: "top SKUs by sales"     │
│        • Retrieve 20 SQL examples              │
│                                                │
│    3b. SCHEMA MANAGER                          │
│        • Embedding similarity: top 10 tables   │
│        • Keyword boost: "SKU", "sales"         │
│        • Result: CustomerInvoiceDetail, Sku    │
│                                                │
│    3c. IMPROVED SQL GENERATOR                  │
│        • Inject retrieved SQL as conversation  │
│        • Follow example patterns strictly      │
│        • Generate SQL using focused schema     │
│                                                │
│    3d. EXECUTE & VALIDATE                      │
│        • Run SQL → DataFrame                   │
│        • If error → SQL Exception Agent        │
│        • Cache result in Redis                 │
│                                                │
│    Output: DataFrame with top 5 SKUs           │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ 4. EXECUTE STEP 2: VISUALIZATION AGENT         │
│    • Load DataFrame from Step 1                │
│    • Detect chart type: bar chart              │
│    • Generate Plotly visualization             │
│    • Save as HTML file                         │
│    • Return: visualization path                │
└────────────────┬───────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────┐
│ 5. SEND RESULTS VIA WEBSOCKET                  │
│    Message 1: TYPE_TABLE (DataFrame JSON)      │
│    Message 2: TYPE_GRAPH (HTML path)           │
│    Message 3: TYPE_SUMMARY (text summary)      │
└────────────────────────────────────────────────┘
```

---

## Core Functionalities

### 1️⃣ **Natural Language Query Processing**

**Input**: Raw user questions in English  
**Output**: Structured SQL queries with results

**Example**:
```
Query: "Which customers visited last month but not this month?"

Processing:
1. Enrich: "last month" → September 2025, "this month" → October 2025
2. Verify: Check if "customers", "visited" entities exist
3. Generate SQL:
   SELECT DISTINCT c.customer_name 
   FROM visits_per_customer c
   WHERE DATE_TRUNC('month', c.visit_date) = '2025-09-01'
   AND c.customer_id NOT IN (
     SELECT customer_id FROM visits_per_customer
     WHERE DATE_TRUNC('month', visit_date) = '2025-10-01'
   )
4. Execute and return results
```

**Features**:
- Date/time parsing (last month, Q1 2025, yesterday)
- Entity resolution (brand names, SKU codes, customer names)
- Ambiguity detection (ask for clarification if unclear)
- Follow-up query handling (using conversation context)

---

### 2️⃣ **Multi-Step Query Decomposition**

**Purpose**: Break complex queries into sequential tasks

**Example**:
```
Query: "Get top 3 customers by sales in Mumbai, 
        analyze their purchase patterns, 
        and email the report to manager@company.com"

Decomposition:
Step 1: db_query    → Get top 3 customers in Mumbai
Step 2: summary     → Analyze purchase patterns from Step 1 data
Step 3: email       → Send analysis to manager@company.com

Execution:
├─ Step 1: Execute SQL → Cache result
├─ Step 2: Load cached data → Generate summary → Cache
└─ Step 3: Load summary → Send email → Confirm
```

**Agent-Aware Decomposition**:
- **Heuristic Optimization**: Fast pattern matching for common queries (no LLM call)
- **LLM Analysis**: Deep reasoning for complex queries
- **Capability Matching**: Ensures each step matches agent capabilities
- **Dependency Tracking**: Passes results between steps via `intermediate_results`

---

### 3️⃣ **Conversation Memory & Context Enrichment**

**Storage**: Redis-backed conversation history (1-hour TTL)

**Capabilities**:
```
User: "Show sales for customer ABC123 in September"
System: [Executes query, returns $50,000]

User: "What about October?"
System: [Enriches to "Show sales for customer ABC123 in October"]

User: "Send that data to john@example.com"
System: [Loads cached September data, sends via email]
```

**Memory Structure**:
```python
{
  "session_id": "unique_session",
  "user_id": "user_123",
  "conversation_history": [
    {
      "role": "user",
      "content": "Show sales for ABC123 in September",
      "timestamp": "2025-10-29T10:00:00"
    },
    {
      "role": "assistant", 
      "content": "Found $50,000 in sales",
      "result": {"query_data": [...], "total": 50000},
      "agent_type": "db_query"
    }
  ],
  "cached_results": {
    "query_hash_xyz": {...}  // Last DB query result
  }
}
```

---

### 4️⃣ **SQL Generation with Retrieval Augmentation**

**Pipeline**:
```
1. RETRIEVAL (SQL Retriever Agent)
   ├─ Embed user question
   ├─ Search embeddings.pkl (cosine similarity)
   └─ Retrieve top 20 similar SQL queries

2. SCHEMA SELECTION (Schema Manager)
   ├─ Embedding similarity (top-k tables)
   ├─ Extract tables from retrieved SQL
   ├─ Keyword boosting ("visit" → add visit_* tables)
   └─ Combine into focused schema (15-25 tables)

3. SQL GENERATION (Improved SQL Generator)
   ├─ Inject retrieved SQL as fake conversation
   ├─ Order by similarity (most similar = last message)
   ├─ Provide focused schema context
   ├─ Strict rules: "MUST use tables from examples"
   └─ Generate SQL following example patterns

4. VALIDATION & EXECUTION
   ├─ Execute SQL → DataFrame
   ├─ If error → SQL Exception Agent (5 iterations)
   │   ├─ Pattern matching (missing table, syntax, etc.)
   │   ├─ Root cause analysis
   │   └─ Generate fix SQL
   └─ Return results
```

**Key Innovation**: 
- Retrieved SQL examples are injected as "conversation history" to guide LLM
- Most similar example appears last (treated as "most recent" context)
- Keyword boosting prevents schema selection from missing domain tables

---

### 5️⃣ **Entity Verification**

**Purpose**: Validate entity names before query execution to prevent failures

**Example**:
```
Query: "Show sales for Coca Cola"

Verification Process:
1. Extract entity: "Coca Cola"
2. Generate verification SQL:
   SELECT brand_name FROM Brand 
   WHERE LOWER(brand_name) LIKE '%coca%cola%'
3. Results: ["Coca-Cola", "Coca Cola Zero"]
4. Ask user: "Did you mean: 1) Coca-Cola  2) Coca Cola Zero?"

User: "1"
→ Update query to use exact name "Coca-Cola"
```

**Skips Verification For**:
- IDs (customer_id=123)
- Partial matches (brand LIKE '%coca%')
- Campaign names (handled separately)
- Geographical locations

---

### 6️⃣ **Data Visualization**

**Auto-Chart Generation**:
```python
Query: "Show monthly sales trend for last 6 months"

Visualization Agent:
1. Receive DataFrame from db_query agent
2. Detect chart type:
   - Time series → Line chart
   - Categorical comparison → Bar chart
   - Distribution → Histogram/Pie chart
3. Generate Plotly chart
4. Save as HTML file (visualizations/*.html)
5. Return path to frontend
```

**Supported Chart Types**:
- Line charts (trends over time)
- Bar charts (categorical comparisons)
- Pie charts (proportions)
- Scatter plots (correlations)
- Heatmaps (2D distributions)
- Multi-series charts (comparing multiple metrics)

---

### 7️⃣ **Error Recovery System**

**SQL Exception Agent** (5-iteration fix attempts):

```
Iteration 1: Execute SQL
   ↓ ERROR: table "CustomerVisit" does not exist
   
Iteration 2: Analyze Error
   ├─ Category: MISSING_TABLE
   ├─ Severity: CRITICAL
   ├─ Root Cause: "Table 'CustomerVisit' not in schema"
   ├─ Fix Strategy: "Use correct table from schema"
   └─ Generate Fix SQL: Use "visits_per_customer" instead
   
Iteration 3: Execute Fixed SQL
   ↓ ERROR: column "dispatchedValue" does not exist
   
Iteration 4: Analyze Error
   ├─ Category: MISSING_COLUMN
   ├─ Fix: Use "dispatchedvalue" (lowercase)
   └─ Generate Fix SQL
   
Iteration 5: Execute
   ✅ SUCCESS → Return results
```

**Error Categories**:
- MISSING_TABLE: Table doesn't exist in schema
- MISSING_COLUMN: Column doesn't exist in table
- SYNTAX_ERROR: SQL syntax mistakes
- DATE_FORMAT: Date parsing issues
- AGGREGATION: GROUP BY / aggregate function errors
- JOIN_ERROR: Join clause problems

---

## Agent Ecosystem

### 🤖 Agent Registry

| Agent | Purpose | Input | Output | Dependencies |
|-------|---------|-------|--------|--------------|
| **Enrich Agent** | Query understanding & enrichment | Raw query + history | Enriched query / Follow-up / Answer | Redis, Schema Manager |
| **Agent-Aware Decomposer** | Multi-step detection | Enriched query | Task sequence + agent plan | Agent Registry |
| **DB Query Agent** | SQL orchestration | Query | DataFrame results | SQL Generator, Retriever, Exception Agent |
| **SQL Retriever Agent** | Similar SQL search | Query | Top-20 similar SQLs | embeddings.pkl |
| **Improved SQL Generator** | SQL generation | Query + context | SQL query | Schema Manager, Retriever |
| **SQL Exception Agent** | Error fixing | SQL + error | Fixed SQL | Schema Manager |
| **Schema Manager** | Focused schema selection | Query + retrieved SQL | Relevant tables (15-25) | CubeJS, Embeddings |
| **Entity Verification Agent** | Entity validation | Query | Verified entities / Clarification | Database |
| **Visualization Agent** | Chart generation | DataFrame + query | HTML chart file | Plotly |
| **Summary Agent** | Data analysis | DataFrame + query | Text insights | LLM |
| **Email Agent** | Send emails | Recipients + content | Confirmation | SMTP |
| **Meeting Scheduler Agent** | Schedule meetings | User ID + date | Calendar entry | File system |
| **Campaign Agent** | Campaign queries | Campaign name | Campaign data / Redirect | Database |

---

### 🔗 Agent Capabilities Matrix

```
┌─────────────────────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ Capability          │ DB   │ Viz  │ Sum  │Email │Meet  │Camp  │
├─────────────────────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ Query Database      │  ✅  │  ❌  │  ❌  │  ❌  │  ❌  │  ⚠️  │
│ Aggregate Data      │  ✅  │  ❌  │  ❌  │  ❌  │  ❌  │  ❌  │
│ Create Charts       │  ❌  │  ✅  │  ❌  │  ❌  │  ❌  │  ❌  │
│ Generate Insights   │  ❌  │  ❌  │  ✅  │  ❌  │  ❌  │  ❌  │
│ Send Emails         │  ❌  │  ❌  │  ❌  │  ✅  │  ❌  │  ❌  │
│ Schedule Meetings   │  ❌  │  ❌  │  ❌  │  ❌  │  ✅  │  ❌  │
│ Use Cached Data     │  ✅  │  ✅  │  ✅  │  ✅  │  ❌  │  ❌  │
│ Multi-step Support  │  ✅  │  ✅  │  ✅  │  ✅  │  ✅  │  ❌  │
└─────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┘

Legend:
✅ Full support
⚠️ Partial (may redirect)
❌ Not supported
```

---

## Data Flow

### 📊 Message Flow (WebSocket)

```
CLIENT                   WEBSOCKET MANAGER           ORCHESTRATOR
  │                             │                          │
  │ Connect (session_token)     │                          │
  ├─────────────────────────────>                          │
  │                             │ Validate token           │
  │                             │ Initialize session       │
  │                             │ Load UserContext         │
  │                             <─────────────────────────┐│
  │ Connection Confirmed        │                         ││
  <─────────────────────────────┤                         ││
  │                             │                         ││
  │ {"query": "Show sales"}     │                         ││
  ├─────────────────────────────>                         ││
  │                             │ Process query            ││
  │                             ├─────────────────────────>││
  │                             │                         ││
  │                             │ Stream responses:       ││
  │                             │   TYPE_PROCESSING       ││
  │ ← Processing...             <─────────────────────────┤│
  │                             │   TYPE_TABLE            ││
  │ ← Table Data (JSON)         <─────────────────────────┤│
  │                             │   TYPE_SUMMARY          ││
  │ ← Summary Text              <─────────────────────────┤│
  │                             │   TYPE_COMPLETE         ││
  │ ← Completion                <─────────────────────────┘│
  │                             │                          │
```

### 🔄 State Machine Flow (LangGraph)

```
START
  │
  ▼
┌─────────────────┐
│ INITIALIZE      │  • Set session_id, user_id
│                 │  • Load conversation history
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ENRICH          │  • EnrichAgent.enrich_query()
│                 │  • Returns: complete_question / follow_up / answer
└────────┬────────┘
         │
    ┌────▼────┐
    │Decision │
    └────┬────┘
         │
    ┌────┼─────┬───────────┐
    │    │     │           │
    ▼    ▼     ▼           ▼
 follow  ans  complete   error
   _up   wer  _question
    │    │       │
    │    │       ▼
    │    │  ┌─────────────────┐
    │    │  │ DECOMPOSE       │  • AgentAwareDecomposer
    │    │  │                 │  • Single-step or multi-step?
    │    │  └────────┬────────┘
    │    │           │
    │    │      ┌────▼─────┐
    │    │      │Decision  │
    │    │      └────┬─────┘
    │    │           │
    │    │      ┌────┼────┐
    │    │      │    │    │
    │    │      ▼    ▼    ▼
    │    │    single multi error
    │    │     step  step
    │    │      │     │
    │    │      │     ▼
    │    │      │  ┌─────────────────┐
    │    │      │  │ MULTI-STEP      │  • Execute tasks sequentially
    │    │      │  │ THINKING AGENT  │  • Pass results via
    │    │      │  │                 │    intermediate_results
    │    │      │  └────────┬────────┘
    │    │      │           │
    │    │      │      ┌────▼─────┐
    │    │      │      │ ROUTE TO │
    │    │      │      │  AGENT   │
    │    │      │      └────┬─────┘
    │    │      │           │
    │    │      │      [Agent executes]
    │    │      │           │
    │    │      │      ┌────▼─────┐
    │    │      │      │ Check if │
    │    │      │      │ more     │
    │    │      │      │ steps?   │
    │    │      │      └────┬─────┘
    │    │      │           │
    │    │      │      ┌────┼────┐
    │    │      │      │         │
    │    │      │     Yes       No
    │    │      │      │         │
    │    │      │      └─►THINKING AGENT
    │    │      │                │
    │    │      ▼                │
    │    │  ┌─────────────────┐ │
    │    │  │ CLASSIFY        │ │
    │    │  │                 │ │
    │    │  │ • Determine     │ │
    │    │  │   agent type    │ │
    │    │  └────────┬────────┘ │
    │    │           │          │
    │    │           ▼          │
    │    │  ┌─────────────────┐ │
    │    │  │ ROUTE_TO_AGENT  │ │
    │    │  │                 │ │
    │    │  │ • Execute agent │ │
    │    │  └────────┬────────┘ │
    │    │           │          │
    │    └───────────┼──────────┘
    │                │
    └────────────────┼──────────┐
                     │          │
                     ▼          ▼
                  ┌─────────────────┐
                  │ COMPLETED       │  • Format final result
                  │                 │  • Send via WebSocket
                  └────────┬────────┘
                           │
                           ▼
                          END
```

---

## Key Components

### 1. **User Context Management**

**UserContext Class**: Centralized user session state
```python
class UserContext:
    user_id: str
    user_name: str
    email: str
    schema_map: Dict[str, str]           # Table → Schema
    embeddings_schema: np.ndarray        # Table embeddings
    schema_list: List[str]               # All table schemas
    cubejs_data: Dict                    # CubeJS metadata
    auth_token: str                      # CubeJS auth token
```

**Loading Process**:
1. Decode base64 token → Extract auth_token + user_id
2. Fetch CubeJS metadata (column descriptions)
3. Query INFORMATION_SCHEMA (table/column structure)
4. Combine into formatted schema strings
5. Generate embeddings for schema (once per user)
6. Cache in memory (reused across queries)

---

### 2. **Schema Manager**

**Focused Schema Selection** (Prevents token overflow):

```
Problem: Full schema = 100+ tables = 50K+ tokens
Solution: Select only 15-25 relevant tables

Algorithm:
1. Embedding Similarity
   ├─ Embed user query
   ├─ Compute cosine similarity with all table schemas
   └─ Select top-k (k=10)

2. Retrieved SQL Tables
   ├─ Extract table names from retrieved SQL queries
   └─ Add those tables to selection

3. Keyword Boosting (NEW)
   ├─ If query contains "visit" → add all *visit* tables
   ├─ If query contains "inventory" → add all *inventory* tables
   └─ Prevents missing domain-specific tables

Result: Focused schema with 15-25 tables (5K-10K tokens)
```

---

### 3. **SQL Retriever Agent**

**Embedding-based SQL Search**:

```
Offline (One-time):
1. Collect successful SQL queries from production
2. Embed question-SQL pairs using OpenAI embeddings
3. Save to embeddings.pkl (NumPy array)

Online (Per Query):
1. Embed user question
2. Compute cosine similarity with all stored questions
3. Return top-20 most similar SQL queries
4. Sort by similarity (ascending → most similar last)
5. Inject as fake conversation history to LLM

LLM sees:
  User: "Show customer sales in Mumbai"
  Assistant: "SELECT ... FROM CustomerInvoice ..." (0.65 similarity)
  User: "Get top customers by revenue"
  Assistant: "SELECT ... FROM ViewCustomer ..." (0.78 similarity)
  User: <CURRENT QUERY> (most recent)
```

---

### 4. **Redis Memory Manager**

**Storage Structure**:
```
Keys:
├─ user:{user_id}:session:{session_id}:conversation_history
│  └─ List of {role, content, timestamp, agent_type, result}
│
├─ user:{user_id}:session:{session_id}:last_db_query_result
│  └─ Most recent database query result (DataFrame JSON)
│
├─ user:{user_id}:session:{session_id}:cached_result:{query_hash}
│  └─ Cached result for specific query
│
└─ user:{user_id}:session:{session_id}:enriched_cache:{query_hash}
   └─ Cached enriched query

TTL: 3600 seconds (1 hour)
```

**Custom JSON Encoder** (handles non-serializable types):
- pandas DataFrame → `{"_type": "dataframe", "data": [...], "columns": [...]}`
- numpy types → Python float/int
- datetime → ISO format string
- Sets → Lists

---

### 5. **Token Tracking**

**Monitors LLM usage**:
```python
@track_llm_call decorator tracks:
├─ Input tokens
├─ Output tokens
├─ Total cost ($)
├─ Model name
└─ Timestamp

Per-session aggregation:
├─ Total tokens used
├─ Total cost
├─ Average cost per query
└─ Token breakdown by agent
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Orchestration** | LangGraph | 0.6.7 | State machine workflow engine |
| **LLM Framework** | LangChain | 0.3.27 | LLM abstraction & prompts |
| **LLM Provider** | OpenAI | Latest | GPT-4o for reasoning |
| **Database** | PostgreSQL | - | CubeJS backend database |
| **Memory** | Redis | Latest | Session state & caching |
| **API** | FastAPI | Latest | WebSocket server |
| **WebSocket** | uvicorn + websockets | Latest | Real-time communication |
| **Embeddings** | OpenAI text-embedding-3-small | Latest | Semantic search |
| **Visualization** | Plotly + Kaleido | Latest | Interactive charts |
| **Data Processing** | pandas | Latest | DataFrame operations |
| **Logging** | loguru | Latest | Structured logging |

### Dependencies
```
langgraph==0.6.7
langchain==0.3.27
langchain-openai==0.3.33
python-dotenv
scikit-learn              # Cosine similarity
psycopg2-binary          # PostgreSQL driver
pandas                   # Data manipulation
plotly                   # Charts
kaleido                  # Static image export
redis                    # Caching
fastapi                  # WebSocket server
uvicorn[standard]        # ASGI server
loguru                   # Logging
```

---

## Setup & Configuration

### 1. Environment Variables (`.env`)

```bash
# OpenAI API
OPENAI_API_KEY=sk-...

# Database (CubeJS PostgreSQL)
DB_HOST=your-cubedb-host.com
DB_DATABASE=your_database
DB_USER=your_user
DB_PORT=5432
DB_PASSWORD=fallback_password  # Used if no session_id

# CubeJS API
CUBEJS_API_URL=analytics.vwbeatroute.com/api/v1/meta
CUBEJS_API_TOKEN=your_token

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Session
DEFAULT_SESSION_ID=cli_session
DEFAULT_USER_ID=default_user
```

### 2. Installation

```bash
# Clone repository
git clone <repo-url>
cd LangGraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p visualizations logs user_files
```

### 3. Running the System

**WebSocket Server** (Production):
```bash
python main_websocket.py
# Starts FastAPI server on port 8000
# WebSocket endpoint: ws://localhost:8000/ws/{session_token}
```

**CLI Mode** (Development):
```bash
python main.py
# Interactive command-line interface
```

### 4. Initial Setup

**Generate Embeddings** (One-time):
```python
# Run this once to create embeddings.pkl
from sql_retriever_agent import SQLRetrieverAgent

retriever = SQLRetrieverAgent(llm, embedding_file="embeddings.pkl")
# Manually add SQL queries or import from production logs
```

**Test Connection**:
```python
from db_connection import get_database_connection, test_connection

db = get_database_connection(session_id="test")
result = db.test_connection()
print(result)  # Should show: success=True, db_version=...
```

---

## Key Design Patterns

### 1. **State Machine Pattern** (LangGraph)
- Declarative workflow definition
- Conditional routing based on state
- Automatic state persistence
- Error handling at each node

### 2. **Agent-based Architecture**
- Specialized agents for specific tasks
- Composable and extensible
- Independent testing and deployment
- Clear responsibility boundaries

### 3. **Retrieval-Augmented Generation (RAG)**
- SQL examples as retrieval corpus
- Embedding-based similarity search
- Context injection via conversation history
- Reduces hallucination, improves accuracy

### 4. **Caching Strategy**
- Query result caching (Redis)
- Schema caching (in-memory per user)
- Enriched query caching
- Result deduplication

### 5. **Error Recovery Pattern**
- Iterative fix attempts (max 5)
- Pattern-based error classification
- Root cause analysis
- Automatic retry with fixes

---

## Performance Characteristics

### Latency Breakdown
```
Typical Query: "Show top 10 customers by sales"

├─ WebSocket connection: ~100ms
├─ Enrich Agent: ~500ms (LLM call)
├─ Decomposition: ~50ms (heuristic) or ~800ms (LLM)
├─ Classification: ~200ms
└─ DB Query Agent:
    ├─ SQL Retrieval: ~300ms (embedding search)
    ├─ Schema Selection: ~200ms (similarity + keyword boost)
    ├─ SQL Generation: ~1500ms (LLM with context)
    ├─ SQL Execution: ~500ms (database query)
    └─ Result Formatting: ~100ms

Total: ~3.0-3.5 seconds (first query)
       ~1.5-2.0 seconds (cached enrichment)
       ~0.5-1.0 seconds (fully cached result)
```

### Scalability
- **Concurrent Users**: Limited by Redis and PostgreSQL connection pools
- **Memory Usage**: ~200MB base + ~50MB per active user (schema cache)
- **Token Usage**: ~3000-8000 tokens per complex query
- **Cost**: ~$0.02-0.05 per query (GPT-4o pricing)

---

## Monitoring & Logging

### Log Levels (loguru)
```
INFO  → Agent flow, query progress
DEBUG → Detailed state transitions
WARN  → Recoverable errors, cache misses
ERROR → Critical failures, LLM errors
```

### Key Metrics Logged
- Query processing time
- Token usage per agent
- Cache hit/miss rates
- SQL execution time
- Error recovery attempts
- Multi-step workflow progress

---

## Future Enhancements

### Planned Features
1. **Streaming SQL Generation**: Return partial results as SQL executes
2. **Query Optimization**: Automatic index suggestions
3. **Natural Language Explanations**: Explain SQL in plain English
4. **Multi-tenant Support**: Isolated schemas per organization
5. **Advanced Analytics**: Predictive queries, forecasting
6. **Voice Input**: Speech-to-SQL via Whisper API
7. **Collaborative Queries**: Shared sessions, query templates

---

## Troubleshooting

### Common Issues

**1. "Redis connection failed"**
```bash
# Check Redis is running
redis-cli ping  # Should return: PONG

# Start Redis if needed
redis-server
```

**2. "Database connection pool not initialized"**
```
Check .env variables:
- DB_HOST, DB_DATABASE, DB_USER, DB_PORT
- Ensure session_id (password) is valid
```

**3. "Schema embeddings not found"**
```python
# Regenerate embeddings
user_context.load_schema_from_token(
    base64_token=session_id,
    generate_embeddings=True  # Force regeneration
)
```

**4. "SQL Exception Agent fails with f-string error"**
```
FIXED in latest version:
- All dictionary values extracted before f-strings
- See: sql_exception_agent.py lines 454-456
```

---

## Contact & Support

**Repository**: Query_file_access  
**Owner**: Arunprabhakaran04  
**Branch**: main  
**Last Updated**: October 29, 2025

For issues or questions, please open a GitHub issue.

---

**END OF DOCUMENTATION**
