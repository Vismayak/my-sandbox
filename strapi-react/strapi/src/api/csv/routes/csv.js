'use strict';

/**
 * csv router
 */

const { createCoreRouter } = require('@strapi/strapi').factories;

module.exports = createCoreRouter('api::csv.csv');
