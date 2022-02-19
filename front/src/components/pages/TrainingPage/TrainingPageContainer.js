import React, {useContext} from 'react';
import VerticalStepper from "./VerticalStepper";
import TrainingContextWrapper, {TrainingContext} from "../../../contexts/TrainingContextWrapper";
import TrainingPage from "./TrainingPage";


function TrainingPageContainer(props){
    
    return(
        <TrainingContextWrapper>
            <TrainingPage/>
        </TrainingContextWrapper>
    )
}

export default TrainingPageContainer;