document.addEventListener('DOMContentLoaded', () => {
    const contributorsDiv = document.querySelector('.contributors-table');
    const contributorsTable = document.createElement('table');
    const tbody = document.createElement('tbody');
    // Get all <td> elements
    const allContributors = Array.from(contributorsDiv.querySelectorAll('td'));
    const rows = [];
    while (allContributors.length) {
        const row = allContributors.splice(0, 5); // 5 columns per row
        rows.push(row);
    }
    rows.forEach(row => {
        const tr = document.createElement('tr');
        row.forEach(td => tr.appendChild(td));
        tbody.appendChild(tr);
    });
    // Replace existing content with the new table
    contributorsDiv.innerHTML = '';
    contributorsTable.appendChild(tbody);
    contributorsDiv.appendChild(contributorsTable);
});
