document.addEventListener('DOMContentLoaded', () => {
    const contributorsDiv = document.querySelector('.contributors-table');
    if (!contributorsDiv) return;

    // Collect all contributor <td> elements across all tables in the section
    const allContributors = Array.from(contributorsDiv.querySelectorAll('td'));

    // Outer row to mirror ::::{grid} 2 4 5 5 :gutter: 2
    const gridRow = document.createElement('div');
    gridRow.className = 'sd-row sd-row-cols-2 sd-row-cols-sm-4 sd-row-cols-md-5 sd-row-cols-lg-5 sd-g-2 docutils';

    allContributors.forEach(td => {
        const link = td.querySelector('a');
        const img = td.querySelector('img');
        const name = td.querySelector('sub b') || td.querySelector('sub');
        if (!link || !img) return;

        // Column to mirror :::{grid-item-card}
        const col = document.createElement('div');
        col.className = 'sd-col sd-d-flex-row docutils';

        const card = document.createElement('div');
        card.className = 'sd-card sd-sphinx-override sd-w-100 sd-shadow-sm sd-card-hover docutils';

        // Card body with contributor name
        const cardBody = document.createElement('div');
        cardBody.className = 'sd-card-body docutils';
        const cardTitle = document.createElement('div');
        cardTitle.className = 'sd-card-title sd-font-weight-bold docutils';
        cardTitle.textContent = name ? name.textContent : img.alt;
        cardBody.appendChild(cardTitle);

        // Image at the bottom of the card
        const cardImg = document.createElement('img');
        cardImg.className = 'sd-card-img-bottom';
        cardImg.src = img.src;
        cardImg.alt = img.alt;

        // Stretched link covering the whole card
        const stretchedLink = document.createElement('a');
        stretchedLink.href = link.href;
        stretchedLink.className = 'sd-stretched-link sd-hide-link-text reference external';
        const linkSpan = document.createElement('span');
        linkSpan.textContent = link.href;
        stretchedLink.appendChild(linkSpan);

        card.appendChild(cardBody);
        card.appendChild(cardImg);
        card.appendChild(stretchedLink);
        col.appendChild(card);
        gridRow.appendChild(col);
    });

    contributorsDiv.innerHTML = '';
    contributorsDiv.appendChild(gridRow);
});
