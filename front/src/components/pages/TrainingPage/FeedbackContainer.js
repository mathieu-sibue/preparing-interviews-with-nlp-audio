import React, {useEffect, useState} from "react";
import EmotionButtons from "./EmotionButtons";
import Transcript from "./Transcript";
import ValenceArousalGraph from "./ValenceArousalGraph";
import ProsodicMetrics from "./ProsodicMetrics";
import { Grid, Box, Paper, Typography } from "@material-ui/core";


//const testTranscriptPropositions = ['Alors une expérience difficile dans ma vie professionnelle,', "je dirais que l'expérience la plus difficile à laquelle j'ai fait face dans la vie professionnelle fut ma première professionnelle,", 'ma toute première fois dans ma vie professionnelle,', "c'était un stage donc un stage d'été dans une banque dans la finance,", 'ça a été un gros choc parce que je sorte de centrale-Supélec où les journées étaient assez light passé simple ou il y avait beaucoup de vie associative pas mal de vie festive et je faisais beaucoup de musique en la journée.', 'Là,', "je me suis retrouvé en costard dans des grandes tours à Londres à travailler 10h par jour rien faire d'autre de mes journées de faire que travailler de 8h à 20h pour c'était très bof et voilà,", 'il y avait les trans.', 'Entre les deux et puis le reste du temps une heure pour manger le soir,', "je rentrais chez moi et je n'arrivais à rien faire d'autre.", "C'est vraiment un gros choc comment ils sont là-bas,", "ça m'a presque dégoûté du monde du travail mais après j'ai réussi à trouver une réponse à ce problème là et je suis retourné dans la même entreprise dans la même banque à Londres pour un semestre cette fois-ci,", "c'était ton février 2020.", "Donc comme quoi ça ne m'avait pas totalement dégoûté du monde du travail cette première expérience et cette fois-ci,", "j'étais beaucoup plus discipliné dans ma façon de faire donc j'arrivais à avoir la discipline de me lever très tôt le matin pour faire du sport avant le travail et ensuite j'arrive est enchaîné avec la grosse journée de travail sans être trop fatigué,", "donc ça c'était mon rythme en semaine et le weekend,", "j'arrivais à sortir et à Une vie festive et donc à vivre de façon assez joyeuses le plus difficile donc ça a été le choc dans le changement de façon de faire dans le monde du travail et ma réponse.", 'Du coup à cela,', "c'était d'avoir la discipline de m'adapter et de trouver un nouveau mode de vie convenable."];
//const testNlpEmotions = [{'neutrality': 0.5042439699172974, 'sadness': 0.6186875104904175}, {'neutrality': 0.7802883982658386}, {'neutrality': 0.8950905799865723}, {'neutrality': 0.851387619972229}, {'surprise': 0.9189707636833191}, {'neutrality': 0.9471076130867004}, {'anger': 0.3180086314678192, 'neutrality': 0.3213909864425659}, {'neutrality': 0.9313125610351562}, {'neutrality': 0.8970057964324951}, {'neutrality': 0.7506791353225708}, {'surprise': 0.47776830196380615}, {'anger': 0.47883275151252747, 'disgust': 0.5261656641960144}, {'neutrality': 0.9290719032287598}, {'neutrality': 0.4981413185596466}, {'joy': 0.8176459670066833}, {'neutrality': 0.8521944284439087}, {'surprise': 0.9200300574302673}, {'neutrality': 0.937170147895813}, {'joy': 0.8752334117889404}];
//const testKeywords = ['travail', 'façon', 'monde', 'journées', 'vie professionnelle'];


const rgbaColorsToComplete = {
    'anger': 'rgba(255,99,71,',
    'neutrality': 'rgba(211,211,211,',
    'joy': 'rgba(255,215,0,',
    'sadness': 'rgba(30,144,255,',
    'surprise': 'rgba(144,238,144,',
    'disgust': 'rgba(160,82,45,',
    'fear': 'rgba(238, 130, 238,'
}


export default function FeedbackContainer(props) {

    const [transcriptPropositions, setTranscriptPropositions] = useState(props.result.transcript);
    const [prosodicInsights, setProsodicInsights] = useState(props.result.insights[3]);
    const [nlpEmotions, setNlpEmotions] = useState(props.result.insights[0]);
    const [keywords, setKeywords] = useState(props.result.insights[1]);
    const [arousal, setArousal] = useState(props.result.insights[2]);
    const [emotionButtonsActivation, setEmotionButtonsActivation] = useState()

    useEffect(() => {
        setEmotionButtonsActivation(null)
        setTranscriptPropositions(props.result.transcript);
        setProsodicInsights(props.result.insights[3])
        setNlpEmotions(props.result.insights[0]);
        setKeywords(props.result.insights[1]);
        setArousal(props.result.insights[2]);
    }, [props.replayingNumber])


    var emotionLabels = [];
    for (const sentenceEmotions of nlpEmotions) {
        emotionLabels.push(Object.keys(sentenceEmotions));
    }
    emotionLabels = [].concat.apply([], emotionLabels);
    const presentEmotionLabels = [...new Set(emotionLabels)];


    return (
        <div style={{margin:'auto', padding:'5px', paddingTop: '10px'}}>

            <Grid container direction='row' spacing={2}>

                <Grid item container direction='column' sm={9} spacing={2}>

                    <Grid item container direction='row' spacing={2} alignContent='center'>

                        <Grid item xs={9}>
                            <Transcript 
                                rgbaColorsToComplete={rgbaColorsToComplete}
                                transcriptPropositions={transcriptPropositions}
                                keywords={keywords}
                                arousal={arousal}
                                nlpEmotions={nlpEmotions}
                                emotionButtonsActivation={emotionButtonsActivation}
                                videoTime={props.videoTime}
                            />
                        </Grid>

                        <Grid item xs={3} alignContent='center'>
                            <div style={{textAlign:'center'}}>
                                <EmotionButtons 
                                    rgbaColorsToComplete={rgbaColorsToComplete}
                                    presentEmotionLabels={presentEmotionLabels}
                                    emotionButtonsActivation={emotionButtonsActivation}
                                    replayingNumber={props.replayingNumber}
                                    setEmotionButtonsActivation={setEmotionButtonsActivation}
                                />
                            </div>
                            {
                                emotionButtonsActivation?
                                <div 
                                    style={{
                                        marginTop: '5px',
                                        textAlign: 'center',
                                        fontSize: '10px',
                                        color: '#616161'
                                    }}
                                >
                                    Indice de confiance
                                    <Box 
                                        display='flex' 
                                        justifyContent="center" 
                                        style={{marginTop: '5px'}}
                                    >
                                        <Box p={1} bgcolor={rgbaColorsToComplete[emotionButtonsActivation]+'0.25)'}>
                                            {'<0.5'}
                                        </Box>
                                        <Box p={1} bgcolor={rgbaColorsToComplete[emotionButtonsActivation]+'0.625)'}>
                                            {'0.5-0.75'}
                                        </Box>
                                        <Box p={1} bgcolor={rgbaColorsToComplete[emotionButtonsActivation]+'0.875)'} color="white">
                                            {'>0.75'}
                                        </Box>
                                    </Box>                            
                                </div>

                                :
                                <></>
                            }
                        </Grid>

                    </Grid>

                    <Grid item>
                        <ProsodicMetrics
                            prosodicInsights={prosodicInsights}
                        />
                    </Grid>

                </Grid>

                <Grid item sm={3}>
                    <ValenceArousalGraph
                        videoTime={props.videoTime}
                        nlpEmotions={nlpEmotions}
                        arousal={arousal}
                        rgbaColorsToComplete={rgbaColorsToComplete}
                    />
                </Grid>

            </Grid>
            <>
                {props.children}
            </>
        </div>
    )
}
