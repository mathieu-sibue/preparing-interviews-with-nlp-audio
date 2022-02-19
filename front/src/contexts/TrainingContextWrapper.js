import React, { createContext, useState, useEffect } from "react"
import { CircularProgress } from "@material-ui/core"
import questionRequests from "../APIrequests/questionRequests";


export const TrainingContext = createContext();

export default function TrainingContextWrapper({ children }) {


    const [isLoaded, setIsLoaded] = useState(false);
    const [questionsAndResults, setQuestionsAndResults] = useState(null)

    useEffect(() => {
        async function retrieveQuestions(){
            let storedQuestionsAndResults = JSON.parse(localStorage.getItem('questionsAndResults'));
            if (storedQuestionsAndResults !== null){
                setQuestionsAndResults(storedQuestionsAndResults)
            }
            else{
                const results = await questionRequests.getQuestions()
                console.log(results)
                let questionsAndResults = []
                for (let k=0; k< results.length ; k++){
                    questionsAndResults.push({
                        question:{
                            id : results[k]._id,
                            text : results[k].question
                        },
                        results: [],
                        videos: []
                    })
                }
                setQuestionsAndResults(questionsAndResults)
            }

        }
        retrieveQuestions().then(
            () => setIsLoaded(true)
        )
    }, [])

    useEffect(() => {
        async function onUnload() {
            if (questionsAndResults !== null) {
                const currentQuestionsAndResults = questionsAndResults;
                localStorage.setItem('questionsAndResults', JSON.stringify(currentQuestionsAndResults));
            }
        }
        window.addEventListener("beforeunload", onUnload)
        return () => window.removeEventListener("beforeunload", onUnload)
    })

    return (
        <div>
            {
                isLoaded?
                    <TrainingContext.Provider value={{
                        questionsAndResults,
                        setQuestionsAndResults,
                    }}>
                        {children}
                    </TrainingContext.Provider>:

                    <div style={{display:'flex', justifyContent:'center', alignItems:'center', marginTop: "50vh", transform: "translateY(-50%)"}}>
                        <CircularProgress disableShrink style={{margin: "auto"}}/>
                    </div>
            }
        </div>

    )
}