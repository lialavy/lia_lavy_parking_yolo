
async function fetchData() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        
        document.getElementById('occupied').innerText = data.occupied;
        document.getElementById('available').innerText = data.available;
        document.getElementById('last_updated').innerText = data.last_updated;
    } catch (error) {
        console.error('Error fetching parking status:', error);
    }
}

// Fetch immediately and then every second
fetchData();
setInterval(fetchData, 1000);
