"""Compact nlp_flow model components."""

from newera.nlp_flow.model.components.architecture.foundation.blocks import CompactDecoderBlock, RMSNorm
from newera.nlp_flow.model.components.training.objectives.cfm import TwoStageASTCFM, TwoStageFlowOutput
from newera.nlp_flow.model.components.architecture.representations.channels import TypedChannelMemory, TypedChannelMemoryEncoder
from newera.nlp_flow.model.components.experts.competence.competence import (
    CompetenceCell,
    UnifiedCompetenceBank,
    UnifiedCompetenceConfig,
    UnifiedCompetenceOutput,
    UnifiedPlanner,
    UnifiedPlannerOutput,
)
from newera.nlp_flow.model.components.architecture.foundation.config import NLPModelConfig
from newera.nlp_flow.model.components.tasks.conversation.conversation import (
    CODE_LANGUAGES,
    REPLY_MODES,
    TURN_TYPES,
    VALIDITY_TYPES,
    ConversationControlConfig,
    ConversationControlHead,
    TurnSegmentEmbedding,
)
from newera.nlp_flow.model.components.experts.banks.expert_bank import (
    ExpertBankConfig,
    ExpertBankOutput,
    ExpertResidualBank,
    FixedExpertPool,
)
from newera.nlp_flow.model.components.experts.banks.expert_factory import (
    ContextConditionedExpertFactory,
    ExpertBlueprint,
    GeneratedExpertState,
)
from newera.nlp_flow.model.components.architecture.representations.fusion import TypedChannelFusion, TypedFusionOutput
from newera.nlp_flow.model.components.experts.ensemble.gap_ensemble import (
    GAP_NAMES,
    GapEnsemble,
    GapEnsembleConfig,
    GapEnsembleOutput,
)
from newera.nlp_flow.model.components.tasks.output.heads import TypedChannelHeads
from newera.nlp_flow.model.components.tasks.output.json_grammar import (
    JSONActionHead,
    JSONGrammarState,
    JSONSchemaKeyHead,
    JSONTypedHeads,
    SchemaGrammarState,
    SchemaNode,
    json_action_loss,
    schema_key_features,
    schema_key_loss,
    validate_json_action_sequence,
)
from newera.nlp_flow.model.components.architecture.foundation.model import CompactNLPDecoder, NLPModelOutput
from newera.nlp_flow.model.components.experts.banks.neuron_module import NeuronModule, NeuronModuleOutput, NeuronState
from newera.nlp_flow.model.components.experts.competence.runtime_competence import (
    GAP_EVIDENCE_FIELDS,
    RUNTIME_STORE_VERSION,
    RuntimeCompetenceConfig,
    RuntimeCompetenceManager,
    RuntimeCompetenceRecord,
    RuntimeGapDetector,
)
from newera.nlp_flow.model.components.training.scopes.training_scopes import (
    ScopeController,
    TrainingScope,
    apply_training_scope,
    frozen_parameter_delta,
    parameter_snapshot,
)

__all__ = [
    "CODE_LANGUAGES",
    "GAP_EVIDENCE_FIELDS",
    "GAP_NAMES",
    "REPLY_MODES",
    "RUNTIME_STORE_VERSION",
    "TURN_TYPES",
    "VALIDITY_TYPES",
    "CompactDecoderBlock",
    "CompactNLPDecoder",
    "CompetenceCell",
    "ContextConditionedExpertFactory",
    "ConversationControlConfig",
    "ConversationControlHead",
    "ExpertBankConfig",
    "ExpertBankOutput",
    "ExpertBlueprint",
    "ExpertResidualBank",
    "FixedExpertPool",
    "GapEnsemble",
    "GapEnsembleConfig",
    "GapEnsembleOutput",
    "GeneratedExpertState",
    "JSONActionHead",
    "JSONGrammarState",
    "JSONSchemaKeyHead",
    "JSONTypedHeads",
    "NLPModelConfig",
    "NLPModelOutput",
    "NeuronModule",
    "NeuronModuleOutput",
    "NeuronState",
    "RMSNorm",
    "RuntimeCompetenceConfig",
    "RuntimeCompetenceManager",
    "RuntimeCompetenceRecord",
    "RuntimeGapDetector",
    "SchemaGrammarState",
    "SchemaNode",
    "ScopeController",
    "TrainingScope",
    "TurnSegmentEmbedding",
    "TwoStageASTCFM",
    "TwoStageFlowOutput",
    "TypedChannelFusion",
    "TypedChannelHeads",
    "TypedChannelMemory",
    "TypedChannelMemoryEncoder",
    "TypedFusionOutput",
    "UnifiedCompetenceBank",
    "UnifiedCompetenceConfig",
    "UnifiedCompetenceOutput",
    "UnifiedPlanner",
    "UnifiedPlannerOutput",
    "apply_training_scope",
    "frozen_parameter_delta",
    "json_action_loss",
    "parameter_snapshot",
    "schema_key_features",
    "schema_key_loss",
    "validate_json_action_sequence",
]
