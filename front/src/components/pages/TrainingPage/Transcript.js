import React, { useState, useEffect } from "react";
import Proposition from "./Proposition";
import { Paper, Typography } from "@material-ui/core";


export default function Transcript(props) {

    const [currentWordIndex, setCurrentWordIndex] = useState(0);
    const [currentPropositionIndex, setCurrentPropositionIndex] = useState(0);

    const transcriptPropositions = props.transcriptPropositions;
    const keywords = props.keywords;
    const arousal = props.arousal;
    const nlpEmotions = props.nlpEmotions;
    const emotionButtonsActivation = props.emotionButtonsActivation;
    const rgbaColorsToComplete = props.rgbaColorsToComplete;
    const videoTime = props.videoTime;


    useEffect(() => {

        const offset = 0.4;   // avg word pronunciation duration
        const lastWordOffset = 0.2;

        if (!videoTime || videoTime === 0) {
            setCurrentWordIndex(0)
            setCurrentPropositionIndex(0)

        } else if (
            videoTime > lastWordOffset + arousal[arousal.length-1][arousal[arousal.length-1].length-1]['start_time']
        ) {
            setCurrentWordIndex(arousal[arousal.length-1].length-1)
            setCurrentPropositionIndex(arousal.length-1)

        } else {
            for (let i = 0; i < nlpEmotions.length; i++) {

                // case where the proposition i considered comes in too early compared to videoTime => we skip directly to the next proposition
                if (
                    i < nlpEmotions.length-1
                    && arousal[i+1].length >= 2
                    && (
                        videoTime > arousal[i+1][1]['start_time'] - offset   // we get the real timestamp of the start of the first word of the next proposition
                    )
                ) {
                    continue;

                // case where we find ourselves between two propositions (between the ith and the (i+1)th, with i+1 not the last prop); the word having the timestamp closest to videoTime will be the last of i
                } else if(
                    i < nlpEmotions.length-1
                    && arousal[i+1].length >= 2
                    && (
                        videoTime > arousal[i+1][0]['start_time'] - offset
                        && videoTime < arousal[i+1][1]['start_time'] - offset
                    )
                ) {
                    setCurrentPropositionIndex(i)
                    setCurrentWordIndex(arousal[i].length-1)
                    break;

                // case where we end up in the middle of a proposition: we thus need to find the word with the closest timestamp to videoTime
                } else {

                    // we get the ordinate of the point (i.e. the energy of the word that is being pronunciated)
                    for (let j = 1; j < arousal[i].length; j++) {

                        if (j === arousal[i].length-1) {        
                            setCurrentPropositionIndex(i)
                            setCurrentWordIndex(j-1)
                            break;          // as soon as we have found the right word, we break

                        } else {

                            if (
                                arousal[i][j]['start_time'] - offset <= videoTime
                                && arousal[i][j+1]['start_time'] - offset > videoTime
                            ) {
                                setCurrentPropositionIndex(i)
                                setCurrentWordIndex(j-1)
                                break;          // as soon as we have found the right word, we break
                            }
                        }
                        
                    }
                    break;  // if we end up here, it is that we have found the word j-1 accentuated in proposition i => no need to iterate further on the propositions

                }
            }

        }


    }, [Math.round((props.videoTime + Number.EPSILON) * 10) / 10])


    return (
        <Paper style={{margin: 'auto', padding: '18px', paddingTop: '12px'}} variant="outlined">
            <Typography align='left' variant="h6" component="h6">
                Transcription
            </Typography>
            <div style={{textAlign: 'justify', fontSize:'13px'}}>
                {
                    transcriptPropositions.map(
                        (proposition, index) => {
                            return (
                                <>
                                    <Proposition
                                        rgbaColorsToComplete={rgbaColorsToComplete}
                                        emotionButtonsActivation={emotionButtonsActivation}
                                        propositionText={proposition}
                                        keywords={keywords}
                                        arousal={arousal}
                                        nlpEmotions={nlpEmotions[index]}
                                        currentWordIndex={currentWordIndex}
                                        currentPropositionIndex={currentPropositionIndex}
                                        mappedPropositionIndex={index}
                                        key={index}
                                    />
                                    {' '}
                                </>
                            )
                        }
                    )
                } 
           
            </div>
            <br/>
            <div style={{fontSize:'11px', fontStyle: 'italic', textAlign: 'left'}}>
                Légende : &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style={{fontStyle:'normal'}}>gras</b> : mot-clé&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<u style={{fontStyle:'normal'}}>souligné</u> : mot accentué
            </div>
        </Paper>
    )
}