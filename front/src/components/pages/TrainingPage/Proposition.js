import React from "react";


export default function Proposition(props) {
    
    const rgbaColorsToComplete = props.rgbaColorsToComplete;
    const propositionText = props.propositionText;
    const keywords = props.keywords;
    const arousal = props.arousal.flat();
    const nlpEmotions = props.nlpEmotions;
    const emotionButtonsActivation = props.emotionButtonsActivation;
    const wordList = propositionText.split(
        /(?<=[ ])(?!$)/g
    );
    const containsEmotionActivatedButton = emotionButtonsActivation && Object.keys(nlpEmotions).includes(emotionButtonsActivation);

    return (
        <span
            style={
                {
                    backgroundColor: containsEmotionActivatedButton?
                    rgbaColorsToComplete[emotionButtonsActivation]+nlpEmotions[emotionButtonsActivation]+')'
                    :
                    'white'
                }
            }
        >
            {
                wordList.map(
                    (word, index) => {

                        if (
                            props.mappedPropositionIndex === props.currentPropositionIndex
                            && index === (
                                props.currentWordIndex
                            )
                        ) {
                            for (const keyword of keywords) {
                                if (word.includes(keyword)) {
                                    return <b><span style={{backgroundColor:'black', color:'white'}}>{word.slice(0, -1)}</span>{word.slice(-1)}</b>
                                }
                            }
                            return <><span style={{backgroundColor:'black', color:'white'}}>{word.slice(0, -1)}</span>{word.slice(-1)}</>
                        }

                        for (const keyword of keywords) {
                            
                            if (word.includes(keyword)) {
                                if (index < arousal.length && arousal[index]['underline']) { 
                                    return <b key={index}><u>{word.slice(0, -1)}</u>{word.slice(-1)}</b>
                                }
                                return <b key={index}>{word}</b>                       
                            }

                            if (index < arousal.length && arousal[index]['underline']) {
                                return <span key={index}><u>{word.slice(0, -1)}</u>{word.slice(-1)}</span>
                            }

                        }

                        return <span key={index}>{word}</span>
                    }
                )
            }
        </span>
    )
}