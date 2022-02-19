import React, {useContext, useEffect, useState} from 'react';
import {Button, Container, Typography} from "@material-ui/core";
import Webcam from 'react-webcam';
import {TrainingContext} from "../../../contexts/TrainingContextWrapper";
import insightRequests from "../../../APIrequests/insightRequests";
import {UserContext} from "../../../contexts/UserContextWrapper";
import Timer from "./Timer";
import Feedback from "./FeedBack";
import FeedbackContainer from "./FeedbackContainer";


function WebcamStreamCapture({questionData, index, children}){
    const webcamRef = React.useRef(null);
    const mediaRecorderRef = React.useRef(null);
    const [capturing, setCapturing] = React.useState(false);
    const [recordedChunks, setRecordedChunks] = React.useState([]);
    const [message, setMessage] = React.useState('')
    const [attempt, setAttempt] = React.useState(0)
    const {user} = useContext(UserContext)
    const [videos, setVideos] = React.useState([])
    const [replaying, setReplaying] = React.useState(false)
    const [replayingNumber, setReplayingNumber] = React.useState(0)
    const [computing, setComputing] = React.useState(false)
    const { questionsAndResults, setQuestionsAndResults } = useContext(TrainingContext);
    const [videoTime, setVideoTime] = React.useState(0);


    const handleStartCaptureClick = () => {
        setCapturing(true);
        mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
            mimeType: "video/webm"
        });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();
    };

    const handleDataAvailable =
        ({ data }) => {
            if (data.size > 0) {
                setRecordedChunks((prev) => prev.concat(data), () => {
                });
            }
        }


    const handleStopCaptureClick = () => {
        mediaRecorderRef.current.stop();
    }


    const handleDownload = () => {
        if (recordedChunks.length) {
            const blob = new Blob(recordedChunks, {
                type: "video/webm"
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            document.body.appendChild(a);
            a.style = "display: none";
            a.href = url;
            a.download = "react-webcam-stream-capture.webm";
            a.click();
            window.URL.revokeObjectURL(url);
            setRecordedChunks([]);
        }
    }

    const handleAddingPath = async () => {
        setCapturing(false);
        setComputing(true);
        if (recordedChunks.length) {
            console.log(recordedChunks)
            const blob = new Blob(recordedChunks, {
                type: "video/webm"
            });

            const url = URL.createObjectURL(blob);

            const date = Date.now();

            insightRequests.postVideo(blob, questionData.question.id.toString()+"_"+user.username+"_"+date, questionData.question.text).then(
                (async results => {
                    const resObj = {
                        transcript: results.transcript,
                        insights: results.insights
                    }

                    setQuestionsAndResults(prev => {
                        prev[index].results.push(resObj);
                        prev[index].videos.push(url);
                        return prev
                    })

                    setRecordedChunks([])
                    console.log(questionsAndResults)
                    setReplayingNumber(questionsAndResults[index].videos.length-1);
                    setReplaying(true);
                    setComputing(false);
                })
            )
        }
    }

    useEffect(() => {
        if (recordedChunks.length>0){
            handleAddingPath()
        }
    }, [recordedChunks])

    useEffect(() => {
        if (questionsAndResults[index].videos.length !== 0 && computing===false) {
            setAttempt(questionsAndResults[index].videos.length)
            setCapturing(false)
            setReplaying(true)
        }
    }, [questionsAndResults[index].videos, computing])

    return (
        <>
            {replaying? (
                <div style={{marginTop : 5, textAlign:'center'}}>
                    <Typography align="center" gutterBottom>Tentative n°{replayingNumber+1}</Typography>
                    <video 
                        onTimeUpdate={event => setVideoTime(event.target.currentTime)} controls height={window.innerHeight/2} 
                        src={questionsAndResults[index].videos[replayingNumber]} 
                        type="video/webm"
                    />
                    <div style={{textAlign:'center', marginTop: '5px'}}>
                        <Button 
                            variant="outlined"
                            style={{marginRight:'3px'}} 
                            color="secondary" 
                            disabled={(replayingNumber<=0)} 
                            onClick={() => {setReplayingNumber(replayingNumber-1)}}
                        >
                            Tentative Précédente
                        </Button>
                        <Button 
                            variant="outlined" 
                            style={{marginLeft:'3px'}} 
                            color="secondary" 
                            disabled={(replayingNumber>=questionsAndResults[index].videos.length-1)}
                            onClick={() => {setReplayingNumber(replayingNumber+1)}}
                        >
                            Tentative Suivante
                        </Button>
                    </div>
                </div>
            ) : (
                <div style={{marginTop : 5, margin:"auto", textAlign:'center'}}>
                    <Typography align="center" gutterBottom>Tentative n°{attempt+1}</Typography>
                    <Webcam height={window.innerHeight/2} audio={true} ref={webcamRef} style={{margin:'auto'}}/>
                </div>
            )
            }

            {capturing ? (
                <div style={{marginTop : 5, textAlign:'center'}}>
                    <Button variant="contained"
                            color="secondary" onClick={() => {handleStopCaptureClick()}}>Arrêter l'enregistrement</Button>
                </div>
            ) : (
                <div style={{marginTop:5, textAlign:'center'}}>
                    {(!replaying) &&
                    <Button variant={(attempt ===0)?"contained" : 'outlined'}
                            style={{marginBottom:'5px'}}
                            color="secondary" onClick={handleStartCaptureClick} disabled={(attempt===0 || computing)}>
                        {computing? (<Typography>Traitement...</Typography>):
                            (
                                (attempt === 0) ? (
                                    <><Typography>Enregistrement dans&nbsp;</Typography><Timer timeOut={5} timeOutHandler={() => handleStartCaptureClick()}/></>
                                    ):
                                "Relancer l'enregistrement")}</Button>
                    }
                    {(attempt >= 1) &&
                    <div>
                        {(!capturing && !computing)&&
                        <div style={{marginTop : 0, textAlign:'center'}}>
                            <Button variant={'outlined'} style={{marginRight:5}} color="primary" onClick={() => {setReplaying(!replaying)}}>{(replaying)? "S'enregistrer une nouvelle fois" : 'Revoir ses tentatives'}</Button>
                        </div>
                        }
                        {replaying &&
                        <div style={{marginTop : 10, textAlign:'center'}}>
                            <FeedbackContainer 
                                videoTime={videoTime} 
                                replayingNumber={replayingNumber} 
                                result={questionsAndResults[index].results[replayingNumber]}
                            >
                                {children}
                            </FeedbackContainer>
                        </div>
                        }
                    </div>
                    }
                </div>
            )}
        </>
    );
};


function Question({questionData, children, ...props}){


    return(
        <Container>
            <Typography variant="h4" component="h4" align="center">
                {questionData.question.text}
            </Typography>
            <WebcamStreamCapture
                index = {props.number-1}
                questionData = {questionData}
                children = {children}
            />
        </Container>
    )
}

export default Question;


