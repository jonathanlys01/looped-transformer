# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from typing import Optional

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from transformers import AutoModelForCausalLM
from typer import Argument, Option
from typing_extensions import Annotated

from looped_gemma import load_hacked_model as load_hacked_gemma
from looped_llama import load_hacked_model as load_hacked_llama


logger = logging.getLogger(__name__)

HELP_PANEL_NAME_1 = "Common Parameters"
HELP_PANEL_NAME_2 = "Logging Parameters"
HELP_PANEL_NAME_3 = "Debug Parameters"
HELP_PANEL_NAME_4 = "Modeling Parameters"

PLAN = (0, 0, 0)  # default autoplan


def accelerate(  # noqa C901
    # === general ===
    model_id: Annotated[
        str,
        Argument(
            help="Model ID to evalutate",
        ),
    ],
    tasks: Annotated[str, Argument(help="Comma-separated list of tasks to evaluate on.")],
    # === opt ===
    auto_model_id: Annotated[
        str,
        Option(
            help="""Use the auto model for evaluation. This will use the model_id as the auto
            model ID. If set to False, the model_id will be used as the model ID.""",
            rich_help_panel=HELP_PANEL_NAME_4,
        ),
    ] = None,
    plan: Annotated[
        Optional[tuple[int, int, int]],
        Option(
            help="Plan to use for evaluation. Only used if `use_looped_g[emma` is set to True",
            rich_help_panel=HELP_PANEL_NAME_4,
        ),
    ] = PLAN,
    use_looped_model: Annotated[
        bool,
        Option(
            help="Use the looped model for evaluation. This is a hack to use the gemma/llama model with lighteval.",
            rich_help_panel=HELP_PANEL_NAME_4,
        ),
    ] = False,
    interpolation_mode: Annotated[
        str,
        Option(
            help="Interpolation mode for the model. Can be one of 'no', 'ema', 'cached_ema', 'accu'.",
            rich_help_panel=HELP_PANEL_NAME_4,
        ),
    ] = "no",
    # === Common parameters ===
    use_chat_template: Annotated[
        bool,
        Option(help="Use chat template for evaluation.", rich_help_panel=HELP_PANEL_NAME_4),
    ] = False,
    system_prompt: Annotated[
        Optional[str],
        Option(help="Use system prompt for evaluation.", rich_help_panel=HELP_PANEL_NAME_4),
    ] = None,
    dataset_loading_processes: Annotated[
        int,
        Option(help="Number of processes to use for dataset loading.", rich_help_panel=HELP_PANEL_NAME_1),
    ] = 1,
    custom_tasks: Annotated[
        Optional[str],
        Option(help="Path to custom tasks directory.", rich_help_panel=HELP_PANEL_NAME_1),
    ] = None,
    num_fewshot_seeds: Annotated[
        int,
        Option(help="Number of seeds to use for few-shot evaluation.", rich_help_panel=HELP_PANEL_NAME_1),
    ] = 1,
    load_responses_from_details_date_id: Annotated[
        Optional[str],
        Option(help="Load responses from details directory.", rich_help_panel=HELP_PANEL_NAME_1),
    ] = None,
    # === saving ===
    output_dir: Annotated[
        str,
        Option(help="Output directory for evaluation results.", rich_help_panel=HELP_PANEL_NAME_2),
    ] = "results",
    results_path_template: Annotated[
        str | None,
        Option(
            help="Template path for where to save the results, you have access to 3 variables, `output_dir`, `org` and `model`. for example a template can be `'{output_dir}/1234/{org}+{model}'`",  # noqa: E501
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ] = None,
    push_to_hub: Annotated[
        bool,
        Option(help="Push results to the huggingface hub.", rich_help_panel=HELP_PANEL_NAME_2),
    ] = False,
    push_to_tensorboard: Annotated[
        bool,
        Option(help="Push results to tensorboard.", rich_help_panel=HELP_PANEL_NAME_2),
    ] = False,
    public_run: Annotated[
        bool,
        Option(help="Push results and details to a public repo.", rich_help_panel=HELP_PANEL_NAME_2),
    ] = False,
    results_org: Annotated[
        Optional[str],
        Option(help="Organization to push results to.", rich_help_panel=HELP_PANEL_NAME_2),
    ] = None,
    save_details: Annotated[
        bool,
        Option(help="Save detailed, sample per sample, results.", rich_help_panel=HELP_PANEL_NAME_2),
    ] = False,
    wandb: Annotated[
        bool,
        Option(
            help="Push results to wandb. This will only work if you have wandb installed and logged in. We use env variable to configure wandb. see here: https://docs.wandb.ai/guides/track/environment-variables/",  # noqa: E501
            rich_help_panel=HELP_PANEL_NAME_2,
        ),
    ] = False,
    # === debug ===
    max_samples: Annotated[
        Optional[int],
        Option(help="Maximum number of samples to evaluate on.", rich_help_panel=HELP_PANEL_NAME_3),
    ] = None,
    job_id: Annotated[
        int,
        Option(help="Optional job id for future reference.", rich_help_panel=HELP_PANEL_NAME_3),
    ] = 0,
):
    """
    Evaluate models using accelerate and transformers as backend.
    """

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        results_path_template=results_path_template,
        save_details=save_details,
        push_to_hub=push_to_hub,
        push_to_tensorboard=push_to_tensorboard,
        public=public_run,
        hub_results_org=results_org,
        wandb=wandb,
    )
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
    )

    if not use_looped_model:
        plan = None

    model_id_ = model_id if auto_model_id is None else auto_model_id

    if use_looped_model:
        if "gemma" in model_id_.lower():
            _, model = load_hacked_gemma(interpolation_mode=interpolation_mode, model_id=model_id_, plan=plan)
        else:
            _, model = load_hacked_llama(interpolation_mode=interpolation_mode, model_id=model_id_, plan=plan)

    if auto_model_id is not None and not use_looped_model:
        print("Using auto model for evaluation.", auto_model_id)  # standard
        model = AutoModelForCausalLM.from_pretrained(
            auto_model_id,
            torch_dtype="auto",
        )
        model.to("cuda")

    config = TransformersModelConfig(model_name=model_id_, use_chat_template=use_chat_template, compile=False)
    model = TransformersModel.from_model(model=model, config=config, use_chat_template=use_chat_template)

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
    )

    pipeline.evaluate()

    pipeline.show_results()

    results = pipeline.get_results()

    # monkey patch the evaluation tracker config
    pipeline.evaluation_tracker.general_config_logger.generation_parameters = {
        "model_id": model_id,
        "use_looped_model": use_looped_model,
        "interpolation_mode": interpolation_mode,
        "plan": plan,
    }

    pipeline.save_and_push_results()

    return results


if __name__ == "__main__":
    import typer

    typer.run(accelerate)
