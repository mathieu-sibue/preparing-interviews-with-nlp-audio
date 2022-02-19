import React, { useState, useEffect } from "react";
import { Scatter } from 'react-chartjs-2';
import { Typography } from "@material-ui/core";


const valenceDict = {
    'anger': -0.8,
    'neutrality': 0,
    'fear': -0.5,
    'surprise': 0,
    'joy': 1,
    'disgust': -1,
    'sadness': -0.7
};



export default function ValenceArousalGraph(props) {

    const rgbaColorsToComplete = props.rgbaColorsToComplete;

    const [x, setX] = useState(0);
    const [y, setY] = useState(0);
    const [pointColor, setPointColor] = useState('grey');

    const data = {
        labels: ['Scatter'],
        datasets: [
            {
                fill: false,
                pointBorderColor: pointColor,
                pointBackgroundColor: pointColor,
                pointBorderWidth: 1,
                pointRadius: 10,
                pointHitRadius: 10,
                data: [
                  { x: x, y: y },
                ]
            }
        ]
    };

    const computeValence = (nlpEmotionsInProposition) => {
        // nlpEmotions is an array of js objects in which each element i contains all emotions detected for proposition i from the transcript
        let res = 0;
        let totalWeights = 0;
        const confidenceWeight = 0.3;  
    
        for (const emotionDetection in nlpEmotionsInProposition) {
            const weight = (1-confidenceWeight) * 1 + confidenceWeight * (Math.log10(1 + nlpEmotionsInProposition[emotionDetection]) / Math.log10(2));
            totalWeights += weight;
            res += weight * valenceDict[emotionDetection];
        }
    
        return res / totalWeights;
    };

    const computeColor = (nlpEmotionsInProposition) => {

        let currentEmotionsInProposition = Object.keys(nlpEmotionsInProposition)
                    
        if (
            currentEmotionsInProposition.length === 1
            && currentEmotionsInProposition[0] === 'neutrality'
        ) {
            return rgbaColorsToComplete['neutrality'] 
                + nlpEmotionsInProposition['neutrality']
                + ')';
    
        } else {
            currentEmotionsInProposition = currentEmotionsInProposition.filter((value, index, arr) => { 
                return value !== 'neutrality';
            })
            const dominantEmotion = currentEmotionsInProposition.reduce(     // emotion autre que la neutralité en laquelle on a le plus confiance
                (emo1, emo2) => nlpEmotionsInProposition[emo1] > nlpEmotionsInProposition[emo2]? emo1 : emo2
            );
            return rgbaColorsToComplete[dominantEmotion] 
                + nlpEmotionsInProposition[dominantEmotion]
                + ')';
    
        }
    };

    
    useEffect(() => {

        const videoTime = props.videoTime;
        const nlpEmotions = props.nlpEmotions;
        const arousal = props.arousal;  
        const offset = 0.35;   // avg word pronunciation duration
        const lastWordOffset = 0.2;

        if (!videoTime || videoTime === 0) {
            setY(arousal[0][0].arousal)
            setX(computeValence(nlpEmotions[0]))
            setPointColor(computeColor(nlpEmotions[0]))

        } else if (
            videoTime > lastWordOffset + arousal[arousal.length-1][arousal[arousal.length-1].length-1]['start_time']
        ) {
            setY(arousal[arousal.length-1][arousal[arousal.length-1].length-1].arousal)
            setX(computeValence(nlpEmotions[arousal.length-1]))
            setPointColor(computeColor(nlpEmotions[arousal.length-1]))
        
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
                    setY(arousal[i][arousal[i].length-1].arousal)
                    break;


                // case where we end up in the middle of a proposition: we thus need to find the word with the closest timestamp to videoTime
                } else {

                    setX(computeValence(nlpEmotions[i]))
                    setPointColor(computeColor(nlpEmotions[i]))

                    // we get the ordinate of the point (i.e. the energy of the word that is being pronunciated)
                    for (let j = 1; j < arousal[i].length; j++) {

                        if (j === arousal[i].length-1) {        
                            setY(arousal[i][j-1].arousal)
                            break;          // as soon as we have found the right word, we break

                        } else {

                            if (
                                arousal[i][j]['start_time'] - offset <= videoTime
                                && arousal[i][j+1]['start_time'] - offset > videoTime
                            ) {
                                setY(arousal[i][j-1].arousal)
                                break;          // as soon as we have found the right word, we break
                            }
                        }
                        
                    }
                    break;      // if we end up here, it is that we have found the word j-1 accentuated in proposition i => no need to iterate further on the propositions
                }
            }

        }


    }, [Math.round((props.videoTime + Number.EPSILON) * 10) / 10])
    

    return (
        <div style={{width:'300px', height:'300px', textAlign:'center'}}>
            <Typography align='center' variant="h6" component="h6" style={{marginTop: '0', marginBottom: '6px'}}>
                Évolution sur la vidéo
            </Typography>
          
          <Scatter 
            data={data} 
            width={1}
            height={1}
            options={{
                tooltips: {enabled: false},
                hover: {mode: null},
                maintainAspectRatio: false,
                animation: {
                    duration: 75
                },
                legend: {
                    display: false
                },
                scales: {
                    yAxes: [
                        { 
                            ticks: { 
                                min: -1, 
                                max: 1, 
                                maxTicksLimit: 5,
                                stepSize: 0.5
                            },
                            scaleLabel: {
                                display: true,
                                labelString: 'Énergie'
                            }
                        }
                    ],
                    xAxes: [
                        { 
                            ticks: { 
                                min: -1, 
                                max: 1, 
                                maxTicksLimit: 5,
                                stepSize: 0.5
                            },
                            scaleLabel: {
                                display: true,
                                labelString: 'Polarité'
                            }
                        }
                    ]
                }
            }}
          />
        </div>
      );

}