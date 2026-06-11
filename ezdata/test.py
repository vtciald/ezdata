# def _apply_p_correction(
#     self,
#     correction: str,
# ):
    
#     valid_corrections = {'bonferroni', 'holm-bonferroni', 'benjamini-hochberg'}

#     # TODO: add a _validate_p_correction_args() method to handle some of the below?

#     correction = correction.lower()

#     if correction not in valid_corrections:
#         raise ValueError(f'Unrecognized argument for correction: \'{correction}\'. Supported choices: {valid_corrections}.')
    
#     if 'p_value' not in df.columns:
#         raise ValueError(f'DataFrame to _apply_correction must have a column \'p_value\' containing p values.')
    
#     if correction == 'bonferroni':
#         df['p_value'] *= len(df['p_value'])
#         # TODO: Test this correction method.

#     elif correction == 'holm-bonferroni':
#         raise NotImplementedError(f'Correction method \'{correction}\' is not yet implemented.')
#         # TODO: Implement correction method

#     elif correction == 'benjamini-hochberg':
#         raise NotImplementedError(f'Correction method \'{correction}\' is not yet implemented.')
#         # TODO: Implement correction method

#     return df  