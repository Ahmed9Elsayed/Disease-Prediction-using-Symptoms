
def disease_predictor(symptoms_arr,pipeline,df_desc,df_prec):
    """
    Based on patient's symptoms, Predict disease and provide desribtion/precautions for it.
    
    :param symptoms_arr: 1D array, the vector containing symptoms described by the patient
    :param pipeline: model pipeling
    :param df_desc: dataframe containing disease descriptions
    :param df_prec: dataframe containing disease precautions
    :return: void
    """
    import pandas as pd
    prediction = pipeline.predict(symptoms_arr)
    predicted_probs = pipeline.predict_proba(symptoms_arr)
    predicted_probs = pd.Series(predicted_probs.reshape(41,), index = pipeline.classes_).sort_values(ascending = False)
    print('Based on symptoms you provided, You might be experiencing:')
    for i in range(4):
        print(f'{i+1}. {predicted_probs.index[i]} Disease with a chance of: {predicted_probs[i]:.4f}', end = '\n')

    
    print(f'\nYour Condition highly matches: {prediction[0]} Disease, \n\nIf you want description or precautions to be taken for {prediction[0]} Type "describtion" or "precautions", else type "Exit"')
    choice = input()
    while(True):
    
        choice = choice.strip().lower()
        if choice == 'describtion':
            describtion = df_desc.loc[df_desc.index == prediction[0],'Description'][0]
            print(f'\n{prediction[0]}: {describtion}')
            print(f'\nWant some Precautions for {prediction[0]}?')
        elif choice == 'precautions':
            precautions = df_prec.loc[(df_prec.index == prediction[0]),:].values
            print(f'\nDisease: {prediction[0]}')
            for i in range(4):
                print(f' precaution number {i+1}: {precautions[0][i]}')
            print(f'\nWant a describtion for {prediction[0]}?')
        
        elif choice == 'exit':
            break
        else:
            print('Please choose from "describtion", "precautions" or "Exit"')
        choice = input()