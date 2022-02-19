import React, {useContext} from 'react';
import {Redirect} from 'react-router-dom';
import VerticalStepper from "./VerticalStepper";
import TrainingContextWrapper, {TrainingContext} from "../../../contexts/TrainingContextWrapper";
import {UserContext} from "../../../contexts/UserContextWrapper";


function TrainingPage(props){
    const {questionsAndResults} = useContext(TrainingContext);
    const {user} = useContext(UserContext);

    return(
        <div>
        {(user) ? <VerticalStepper questionsData={questionsAndResults}/> : <Redirect to={'/'} />}
        </div>
    )
}

export default TrainingPage;