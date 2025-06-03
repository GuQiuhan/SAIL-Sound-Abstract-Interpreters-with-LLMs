class Step:
    def __init__(self, prompter, composer=None, eos=None, validator=None):
        self.prompter: Callable[[str, Optional[str]], str] = prompter
        self.eos: List[str] = eos or []
        # (prompt, completion, old code) =composer=> new code
        self.composer: Callable[[str, str, str], Union[str, bool]] = composer
        self.validator: Callable[[str], None] = validator  # validate the code
        # add augmentation prompting
        self.aug_prompt = ""
        self.error_generation = ""  # List to store error generation
        self.counter_example = ""  # List to store counter examples
        self.given_code = ""  # provided code

    def set_augmentation_prompt(self, aug_prompt: str, error_generation: str, counter_example: str):
        self.aug_prompt = aug_prompt
        self.error_generation = error_generation
        self.counter_example = counter_example

    def prompter_with_augmentation(self, old_prmpt: str) -> str:
        """Generates the prompt with augmentation, error generation and counter examples."""
        augmented_prompt = old_prmpt
        if self.aug_prompt:
            last_api_index = augmented_prompt.rfind("API: ")
            error_index = augmented_prompt.find("Error Generation:", last_api_index)
            if error_index != -1:
                augmented_prompt = (
                    augmented_prompt[:error_index]
                    + "- "
                    + self.aug_prompt
                    + "\n"
                    + augmented_prompt[error_index:]
                )
            
            last_api_index = augmented_prompt.rfind("API: ")
            generation_index = augmented_prompt.find("Counter Example:", last_api_index)
            if generation_index != -1:
                augmented_prompt = (
                    augmented_prompt[:generation_index]
                    + self.error_generation
                    + "\n"
                    + augmented_prompt[generation_index:]
                )

            last_api_index = augmented_prompt.rfind("API: ")
            ce_index = augmented_prompt.rfind("Generation:", last_api_index)
            if ce_index != -1:
                augmented_prompt = (
                    augmented_prompt[:ce_index]
                    + self.counter_example
                    + "\n"
                    + augmented_prompt[ce_index:]
                )
            
        return augmented_prompt


s= Step(
                            prompter=lambda code: f"""
# DeepPoly DSL Transformer Generation

You are a formal methods expert writing a new transformer rule for a PyTorch operator in the DeepPoly DSL. Below is the abstract domain shape and two existing examples (ReLU and Affine). Now generate a new DSL transformer for the operator below.


API: api
Documentation: doc
Tips:
Error Generation:
Counter Example:
Generation: (Add your transformer below. Only generate the Transformer rule (no comments, no extra output)):
""",
                            composer=None,
                            eos=["\n# END"],
                            validator=None,  # @qiuhan: Constraintflow
                        )
 

s.set_augmentation_prompt("aug_prompt","error_generation","counter_example")
op=s.prompter("")
print(op)
print(s.prompter_with_augmentation(op))

