import React, {useContext} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Stepper from '@material-ui/core/Stepper';
import Step from '@material-ui/core/Step';
import StepLabel from '@material-ui/core/StepLabel';
import StepContent from '@material-ui/core/StepContent';
import Button from '@material-ui/core/Button';
import Paper from '@material-ui/core/Paper';
import Typography from '@material-ui/core/Typography';
import Question from "./Question";
import {TrainingContext} from "../../../contexts/TrainingContextWrapper";
import Feedback from "./FeedBack";


const useStyles = makeStyles((theme) => ({
    root: {
        width: '100%',
        margin: 'auto',
    },
    button: {
        marginTop: theme.spacing(1),
        marginRight: theme.spacing(1),
    },
    actionsContainer: {
        maring: 'auto',
        marginTop: theme.spacing(2),
        marginBottom: theme.spacing(2),
    },
    resetContainer: {
        padding: theme.spacing(3),
    },
}));


export default function VerticalStepper(props) {

    const classes = useStyles();
    const [activeStep, setActiveStep] = React.useState(0);
    const steps = props.questionsData;
    const {responses} = useContext(TrainingContext);

    const handleNext = async () => {
        setActiveStep((prevActiveStep) => prevActiveStep + 1);
    };

    const handleBack = () => {
        setActiveStep((prevActiveStep) => prevActiveStep - 1);
    };

    const handleReset = () => {
        setActiveStep(0);
    };

    return (
        <div className={classes.root}>
            <Stepper activeStep={activeStep} orientation="vertical">
                {steps.map((label, index) => (
                    <Step key={index}>
                        <StepLabel style={{margin:'auto'}}>Question n°{index+1}</StepLabel>
                        <StepContent>
                            <Question questionData={label} number={index+1}>
                            <div className={classes.actionsContainer}>
                                <div>
                                        <Button
                                            variant="contained"
                                            color="primary"
                                            onClick={handleNext}
                                            className={classes.button}
                                        >
                                            {activeStep === steps.length - 1 ? "Accéder au feedback général" : 'Question suivante'}
                                        </Button>
                                </div>
                            </div>
                            </Question>
                        </StepContent>
                    </Step>
                ))}
            </Stepper>
            {activeStep === steps.length && (
                <Paper square elevation={0} className={classes.resetContainer}>
                    <Typography>Vous avez bien répondu à toutes les questions !</Typography>
                    <Button onClick={handleReset} className={classes.button}>
                        Reset
                    </Button>
                </Paper>
            )}
        </div>
    );
}