from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)
from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
################################################################
from aihwkit.simulator.rpu_base import cuda
#########
from transformers import GPT2ForSequenceClassification
import wandb

MODEL_NAME = "gpt2"

def create_ideal_rpu_config(tile_size=512):
    """Create RPU Config with ideal conditions"""
    rpu_config = InferenceRPUConfig(
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=False,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(is_perfect=True),
        noise_model=PCMLikeNoiseModel(prog_noise_scale=0.0, read_noise_scale=0.0, drift_scale=0.0),
        drift_compensation=None,
    )
    return rpu_config
################################################################################
def create_rpu_config(ARGS,modifier_noise, tile_size=512, dac_res=256, adc_res=256):
    """Create RPU Config emulated typical PCM Device"""
    if ARGS.wandb:
        modifier_noise = wandb.config.modifier_noise

    rpu_config = InferenceRPUConfig(
        clip=WeightClipParameter(type=WeightClipType.FIXED_VALUE, fixed_value=1.0),
        modifier=WeightModifierParameter(
            rel_to_actual_wmax=True, type=WeightModifierType.ADD_NORMAL, std_dev=modifier_noise
        ),
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=True,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(
            w_noise_type=WeightNoiseType.PCM_READ,
            w_noise=0.0175,
            inp_res=dac_res,
            out_res=adc_res,
            out_bound=10.0,
            out_noise=0.04,
            bound_management=BoundManagementType.ITERATIVE,
            noise_management=NoiseManagementType.ABS_MAX,
        ),
        noise_model=PCMLikeNoiseModel(),
        drift_compensation=GlobalDriftCompensation(),
    )
    return rpu_config
#######################################################################
def create_model(ARGS,rpu_config,num_classes):
    """Return Question Answering model and whether or not it was loaded from a checkpoint"""

    #model = AutoModelForCausalLM.pretra,ined(MODEL_NAME)#AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    #model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    #GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)
    if not ARGS.digital: 
        model = convert_to_analog(model, rpu_config)
        model.remap_analog_weights()
    if cuda.is_compiled():
        model.cuda()
    print(model)

    return model

def get_model(ARGS,num_classes):
    
    if ARGS.ideal:
        rpu_config = create_ideal_rpu_config()
    else:
        rpu_config = create_rpu_config(ARGS,modifier_noise=ARGS.noise)
    model = create_model(ARGS,rpu_config,num_classes)
    return model
def create_optimizer(model,learning_rate):
    """Create the analog-aware optimizer"""
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)

    optimizer.regroup_param_groups(model)

    return optimizer