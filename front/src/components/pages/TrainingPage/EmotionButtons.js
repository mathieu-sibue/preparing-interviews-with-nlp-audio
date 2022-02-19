import React, {useEffect} from "react";
import ToggleButton from '@material-ui/lab/ToggleButton';
import ToggleButtonGroup from '@material-ui/lab/ToggleButtonGroup';
import { withStyles } from '@material-ui/core/styles';


const StyledToggleButtonGroup = withStyles((theme) => ({
    grouped: {
      margin: theme.spacing(0.5),
      border: 'none',
      '&:not(:first-child)': {
        borderRadius: theme.shape.borderRadius,
      },
      '&:first-child': {
        borderRadius: theme.shape.borderRadius,
      },
    },
}))(ToggleButtonGroup);


const emotionsInFrench = {
    'anger': 'colère',
    'fear': 'peur',
    'joy': 'satisfaction',
    'neutrality': 'neutralité',
    'surprise': 'surprise',
    'sadness': 'tristesse',
    'disgust': 'dégoût'
}


export default function EmotionButtons(props) {

    const presentEmotionLabels = props.presentEmotionLabels;
    const emotionButtonsActivation = props.emotionButtonsActivation;
    const setEmotionButtonsActivation = props.setEmotionButtonsActivation;

    return (
        <StyledToggleButtonGroup
            fullWidth={true}
            orientation="vertical"
            value={emotionButtonsActivation}
            exclusive
            onChange={(event, newLabel) => setEmotionButtonsActivation(newLabel)}
        >
            {
                presentEmotionLabels.map(
                    (label, index) => {
                        return (
                            <ToggleButton 
                                value={label}
                                key={index} 
                                style={{fontSize:'12px'}}
                            >
                                {emotionsInFrench[label]}
                            </ToggleButton>
                        )
                    }
                )
            }
        </StyledToggleButtonGroup>
    );
}