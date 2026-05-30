// SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
// SPDX-License-Identifier: ISC
// hide duplicate toctrees in examples folder and api ref folder
document.addEventListener('DOMContentLoaded', function() {
    const path = window.location.pathname;
    const filename = path.split('/').pop();
    
    if (filename === 'examples_frontpage.html' || 
        filename === 'api_ref_frontpage.html' || 
        path.includes('_collections/tutorials') ||
        path.includes('/api_ref/')) {
        const sidenav = document.querySelectorAll('.bd-toc-item .bd-sidenav');
        sidenav.forEach(function(ul, index) {
            if (index % 2 !== 0) {
                ul.style.display = 'none';
            }
        });
    }
});

// make constants link match all the other class links since its different
setTimeout(function() {
    document.querySelectorAll('.bd-sidenav a, .toctree-wrapper a').forEach(function(link) {
        if (link.href.includes('/utils/constants.html') && !link.querySelector('code')) {
            const text = link.textContent;
            link.textContent = '';
            const code = document.createElement('code');
            code.className = 'docutils literal notranslate';
            const span = document.createElement('span');
            span.className = 'pre';
            span.textContent = text;
            code.appendChild(span);
            link.appendChild(code);
        }
    });
}, 500);