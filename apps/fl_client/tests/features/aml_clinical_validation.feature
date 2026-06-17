Feature: Distributed Clinical Validation for Acute Myeloid Leukaemia (AML)
  As a Data Protection Officer (DPO)
  I want to mathematically guarantee that the synthetic 'Big Pool' is validated locally
  So that data privacy and utility are preserved before catalogue publication.

  Scenario: Successful Edge Hydration and Federated Validation
    Given an active Ubuntu edge node holding a 20% real AML hold-out partition
    And a serialised DCAT-AP Data Contract from the FL Server
    When the Flower gRPC network triggers the set_run_config instruction
    Then the edge node mathematically hydrates the Pydantic DatasetMeta object
    When the server requests distributed synthetic data validation
    Then the node successfully executes the clinical validation plugins
    And generates the HTML privacy and utility reports