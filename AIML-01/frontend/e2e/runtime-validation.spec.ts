import { expect, test } from '@playwright/test';

test('analyst can view and update a persisted Kafka case', async ({ page }) => {
  const apiKey = process.env.VITE_API_KEY ?? 'runtime-validation-api-key';
  const casesResponse = await page.request.get('http://127.0.0.1:8000/cases', {
    headers: { 'X-API-Key': apiKey },
  });
  expect(casesResponse.ok()).toBeTruthy();
  const cases = (await casesResponse.json()).cases.filter(
    (item: { case_status: string }) => item.case_status !== 'Resolved',
  );
  expect(cases.length).toBeGreaterThan(0);
  const transactionId = cases[0].transaction_id;

  await page.goto('/alerts');
  await expect(page.getByText(transactionId)).toBeVisible();

  const card = page
    .locator('div.rounded-2xl')
    .filter({ hasText: transactionId })
    .first();
  const reopen = card.getByRole('button', { name: 'Re-open case' });
  if (await reopen.count()) {
    await reopen.click();
    await expect(card.getByText('Under Review', { exact: true })).toBeVisible();
  }

  await card.getByRole('button', { name: 'Resolve' }).click();
  await expect(card.getByText('Resolved', { exact: true })).toBeVisible();
});

test('analyst health page loads from the authenticated API', async ({ page }) => {
  await page.goto('/health');
  await expect(page.getByText('Healthy', { exact: true })).toBeVisible();
});
