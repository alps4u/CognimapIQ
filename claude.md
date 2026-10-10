## 2. `claude.md`

```markdown
# Claude Development Guide: Senior Health GIS Dashboard

This file serves as the system context and project guidelines for Claude (or any AI assistant) when generating code, reviewing architectures, or assisting with feature development for this repository.

---

## 1. Project Intent & Scope
* **Context:** A web-based GIS dashboard analyzing the relationship between cognitive decline and three primary sensory organ impairments (Vision, Hearing, Mastication).
* **Target Audience:** Public health policy makers, analysts, and health authority decision-makers seeking empirical evidence to shift budgets toward preventative sensory care.
* **Prototype Architecture:** Single-process, serverless Streamlit app using local snapshot data (CDC PLACES in `.parquet` format) and `@st.cache_data`. No standalone backend server or relational database required.

---

## 2. Key Features to Implement
1. **Interactive Spatial Map (Primary View):**
   * Multi-sensory risk index visualization at the US County level (PoC target).
   * WebGL/Pydeck-based choropleth layer with dynamic color scales corresponding to risk scores.
2. **Temporal Slider & Longitudinal View (Secondary View):**
   * Year-by-year time-slider enabling smooth visual transitions of geographical risk over time.
   * Visual policy indicators along the timeline reflecting intervention events.
3. **In-Memory Predictive Analytics Engine:**
   * Composite risk score calculator combining Vision, Hearing, and Mastication (Teeth Loss proxy) variables.
   * Filtering and summary statistics panel for selected health regions.

---

## 3. Engineering & Code Guidelines

### Data Operations
* **Always prefer Parquet for Tabular Data:** Never read raw `.csv` inside Streamlit runtime functions. Convert all input datasets to `.parquet` in pre-processing scripts.
* **Cache Management:** Use `@st.cache_data` for all file loading (`pd.read_parquet`, `gpd.read_file`) and spatial join operations.
* **Variable Mapping:**
  * Vision impairment metric: `VISION`
  * Hearing impairment metric: `HEARING`
  * Mastication proxy metric: `TEETHLOST` (complete tooth loss among seniors $\ge$ 65)
  * Cognitive decline metric: `COGNITION`

### Streamlit & Visualization Standards
* **Performance First:** Minimize map re-renders. Use `pydeck` or `streamlit-folium` with lightweight data bindings.
* **Responsive Layout:** Use `st.sidebar` for filters and control inputs; reserve main content space for maps and key analytics cards (`st.metric`).
* **Clean Code Structure:** Keep business logic, mathematical modeling, and component rendering modularized in separate Python modules (`src/data_loader.py`, `src/model.py`, `src/components.py`).

---

## 4. Primary Development Workflow
When writing or refactoring code for this project, ensure that:
1. No external database connection code (PostgreSQL/SQLAlchemy) is introduced unless explicitly requested for future production migration.
2. All spatial transformations maintain standard coordinate reference systems (`EPSG:4326` for WebGIS compatibility).
3. Public health terminology aligns with policy standards (e.g., "Preventative care", "Intervention window of opportunity", "Composite Risk Score").