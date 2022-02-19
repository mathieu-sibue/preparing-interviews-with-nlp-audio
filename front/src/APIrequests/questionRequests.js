//const burl = "https://#####.mvp.#####.fr"
const burl = "http://localhost:5000"

export default {
    getQuestions : async function(){
        const requestOptions = {
            method: 'GET',
            redirect: 'follow'
            };
    return fetch(burl + "/questions/get", requestOptions)
        .then(response => {
            if(response.ok){
                try {
                    return response.json(); 
                } catch (error) {
                    console.error(error);
                }
            }}
        ).then(response => {
            console.log(response)
            return response
    })
    }
}
