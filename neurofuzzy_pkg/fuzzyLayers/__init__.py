# the same layers for all arcs
from neurofuzzy_pkg.utils.MFs import *
from neurofuzzy_pkg.fuzzyLayers.FuzzificationLayer import FuzzificationLayer
from neurofuzzy_pkg.fuzzyLayers.RuleAntecedentLayer import RuleAntecedentLayer
from neurofuzzy_pkg.fuzzyLayers.RuleConsequentLayer import RuleConsequentLayer

# mamdani specific
from neurofuzzy_pkg.fuzzyLayers.mamdani_layers.DefuzzificationLayerMam import DefuzzificationLayerMam
from neurofuzzy_pkg.fuzzyLayers.mamdani_layers.RuleConsequentLayerMam import RuleConsequentLayerMam

# sugeno specififc
from neurofuzzy_pkg.fuzzyLayers.sugeno_layers.DefuzzificationLayer import DefuzzificationLayer
from neurofuzzy_pkg.fuzzyLayers.sugeno_layers.RuleConsequentLayerSug import RuleConsequentLayerSug
from neurofuzzy_pkg.fuzzyLayers.sugeno_layers.NormalizationLayer import NormalizationLayer
