# Project Architecture Specification (`architecture.md`)

## 1. Executive Summary & Problem Statement

### 1.1 Project Title
Cognimap IQ

### 1.2 Problem Statement
Canada's current senior health care system suffers from three systemic bottlenecks that lead to inefficiencies and increased costs. Provincial spending is heavily focused on terminal care, such as Long-Term Care (LTC) and emergency hospitalization and the current disease-centered healthcare system exponentially increases the cost of neurological care for patients progressing from MCI to progressive dementia. This current model fails to intervene during the "window of opportunity" when low-cost preventative measures are most effective. There are currently active researches regarding the correlation between cognitive decline and sensory organ function. By analyzing this relationship using demographic data, the aim is to visualize relationship between between cognitive decline and sensory organ function and generate indicators and data that can be leveraged into public health policies. The policies to support sensory function decline which can be assessed at a relatively low cost and corrected within a reasonable budget.

### 1.3 Project Objective & System Overview
To design and deploy an interactive web-based GIS dashboard that visualizes predictive risk indicators and spatial correlations between cognitive decline and three primary sensory organ impairments (Vision, Hearing, and Mastication). The system translates demographic and health data into actionable spatial analytics to support evidence-based public health policymaking.

* **Core Visual Views**:
  * **Primary View (Spatial Analysis)**: Interactive choropleth heatmaps illustrating regional risk clusters and spatial overlay between sensory organ decline and cognitive risks at the health region/county level.

  * **Secondary View (Temporal & Policy Impact Analysis)**: A longitudinal time-slider interface enabling users to track temporal trends and progression over time, helping decision-makers evaluate the potential impact and correlation of preventative public health policy interventions.

### 1.4 Target Users
* **Primary Users**:
  * **Public Health Policy Makers, Health Authority Analysts, and Budget Planners**: They will utilize this data-backed tool to shift resource allocation from high-cost, reactive care (LTC, emergency rooms) toward low-cost, preventative sensory health interventions (e.g., subsidized hearing aids, vision screenings, and dental/masticatory care).

* **Ultimate Beneficiaries**:
  * **General Public / Aging Population**: While they do not directly interact with this analytical dashboard, they ultimately benefit from optimized public health service delivery and improved preventative healthcare access.

### 1.5 Data
Due to time constraints in sourcing initial Canadian microdata, this project adopts the CDC PLACES dataset which is available under the Trilemma Foundation catalog as a primary data source for prototype (PoC) development.
Source: https://data.trilemma.foundation/datasets/cdc-places

Once the pipeline and visualization framework are validated with PLACES data, the backend model will be adapted to Canadian longitudinal health datasets (e.g., Canadian Community Health Survey (CCHS) and CLSA), aligning the GIS layers with Canadian Health Regions (Health Authorities)
